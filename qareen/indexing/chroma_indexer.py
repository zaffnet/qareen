"""ChromaDB vector store indexer implementation."""

from __future__ import annotations

import logging
import random
import re
import time
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import chromadb
import requests
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from datasets import DatasetDict
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from PIL import Image, UnidentifiedImageError
from rich.logging import RichHandler
from tqdm import tqdm

from qareen.config.settings import Settings
from qareen.indexing.base import VectorStoreIndexer
from qareen.indexing.exceptions import (
    CollectionNotFoundError,
    InvalidEmbeddingError,
    UnsupportedImageTypeError,
)

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore

    from qareen.dataset.base import DatasetLoader
    from qareen.indexing.models import EmbeddingModel

logger = logging.getLogger(__name__)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)


def setup_logging(rich: bool = True, level: int = logging.INFO) -> None:
    """Configure logging with optional RichHandler.

    Args:
        rich: If True, use RichHandler for formatted output
        level: Logging level (default: INFO)

    """
    if not logging.getLogger().handlers:
        if rich:
            handler = RichHandler(rich_tracebacks=True, show_path=False)
            logging.basicConfig(level=level, format="%(message)s", handlers=[handler])
        else:
            logging.basicConfig(level=level, format="%(message)s")


class EmbeddingModelWrapper(Embeddings):
    """LangChain Embeddings wrapper for EmbeddingModel.

    Wraps the embedding model for use with LangChain's Chroma integration.
    """

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        """Initialize wrapper.

        Args:
            embedding_model: Embedding model instance for query embedding

        """
        self.embedding_model = embedding_model
        self._embedding_dim: int | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents using the embedding model.

        Args:
            texts: Input texts to embed

        Returns:
            List of embedding vectors as lists of floats

        Raises:
            RuntimeError: If embedding dimension cannot be determined or embedding fails
            ValueError: If embedding model returns None for any text

        """
        if not texts:
            return []
        if self._embedding_dim is None:
            try:
                self._embedding_dim = self.embedding_model.embedding_dim
            except (AttributeError, RuntimeError, TypeError) as e:
                logger.exception("Failed to get embedding dimension")
                raise RuntimeError("Cannot determine embedding dimension") from e

        embeddings = []
        for text in texts:
            try:
                embedding = self.embedding_model.embed_text(text)
                if embedding is None:
                    model_id = self.embedding_model.get_model_id()
                    raise ValueError(
                        f"Embedding returned None for text. "
                        f"Model: {model_id}, Text: {text[:100] if text else 'None'}...",
                    )
                embedding_list = cast("list[float]", embedding.tolist())
                if len(embedding_list) != self._embedding_dim:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                        f"got {len(embedding_list)}",
                    )
                embeddings.append(embedding_list)
            except ValueError:
                raise
            except Exception as e:
                logger.exception(f"Failed to embed text: {text[:100] if text else 'None'}...")
                raise RuntimeError("Embedding failed for text") from e

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed query text using the embedding model.

        Args:
            text: Query text to embed

        Returns:
            Text embedding vector as list of floats

        Raises:
            ValueError: If embedding is None or cannot be converted to list

        """
        embedding = self.embedding_model.embed_text(text)
        model_id = self.embedding_model.get_model_id()
        if embedding is None:
            raise ValueError(
                f"Embedding returned None for provided text. "
                f"Model: {model_id}, Text: {text[:100] if text else 'None'}...",
            )
        if hasattr(embedding, "tolist"):
            return cast("list[float]", embedding.tolist())
        if hasattr(embedding, "__iter__") and not isinstance(embedding, (str, bytes)):
            return cast("list[float]", list(embedding))
        raise ValueError(
            f"Unsupported embedding format for model {model_id}. "
            f"Expected array-like object with tolist() method or iterable, "
            f"got {type(embedding).__name__}. Text snippet: {text[:50] if text else 'None'}...",
        )


class ChromaIndexer(VectorStoreIndexer):
    """ChromaDB vector store indexer.

    Handles indexing datasets into ChromaDB with pre-computed multimodal embeddings.
    Supports multiple alpha values and environments.

    Attributes:
        settings: Configuration settings
        dataset_loader: Dataset loader instance
        embedding_model: Embedding model instance

    """

    def __init__(
        self,
        dataset_loader: DatasetLoader,
        embedding_model: EmbeddingModel,
        settings: Settings | None = None,
    ) -> None:
        """Initialize ChromaDB indexer.

        Args:
            dataset_loader: Dataset loader instance
            embedding_model: Embedding model instance
            settings: Configuration settings (uses defaults if not provided)

        """
        self.settings = settings or Settings()
        self.settings.ensure_directories()
        self.dataset_loader = dataset_loader
        self.embedding_model = embedding_model

    def _download_image_with_retry(
        self,
        image_url: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_size_bytes: int = 10 * 1024 * 1024,
    ) -> Image.Image | None:
        """Download image from URL with retry logic and validation.

        Args:
            image_url: URL of the image to download
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            max_size_bytes: Maximum allowed image size in bytes

        Returns:
            PIL Image if successful, None otherwise

        """
        for attempt in range(max_retries):
            try:
                with requests.get(image_url, timeout=30, stream=True) as response:
                    response.raise_for_status()

                    content_type = response.headers.get("Content-Type", "").lower()
                    if not content_type.startswith("image/"):
                        logger.warning(
                            f"Invalid Content-Type '{content_type}' "
                            f"for image URL: {image_url} "
                            f"(attempt {attempt + 1}/{max_retries})",
                        )
                        if attempt == max_retries - 1:
                            return None
                        delay = base_delay * (2**attempt) + random.uniform(0, 0.1 * base_delay)
                        time.sleep(delay)
                        continue

                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        try:
                            size = int(content_length)
                            if size > max_size_bytes:
                                logger.warning(
                                    f"Content-Length {size} exceeds max "
                                    f"{max_size_bytes} bytes for image "
                                    f"URL: {image_url} "
                                    f"(attempt {attempt + 1}/{max_retries})",
                                )
                                if attempt == max_retries - 1:
                                    return None
                                delay = base_delay * (2**attempt) + random.uniform(
                                    0,
                                    0.1 * base_delay,
                                )
                                time.sleep(delay)
                                continue
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Invalid Content-Length header "
                                f"'{content_length}' for image URL: "
                                f"{image_url} "
                                f"(attempt {attempt + 1}/{max_retries}): "
                                f"{e}",
                            )
                            if attempt == max_retries - 1:
                                return None
                            delay = base_delay * (2**attempt) + random.uniform(0, 0.1 * base_delay)
                            time.sleep(delay)
                            continue

                    content = bytearray()
                    for chunk in response.iter_content(chunk_size=8192):
                        if len(content) + len(chunk) > max_size_bytes:
                            logger.warning(
                                f"Image size exceeds max "
                                f"{max_size_bytes} bytes during "
                                f"download for image URL: {image_url} "
                                f"(attempt {attempt + 1}/{max_retries})",
                            )
                            return None
                        content.extend(chunk)

                    if len(content) > 0:
                        try:
                            image_buffer = BytesIO(content)
                            img = Image.open(image_buffer)
                            img.verify()
                            image_buffer.seek(0)
                            image = Image.open(image_buffer)
                            return image
                        except (
                            UnidentifiedImageError,
                            OSError,
                            ValueError,
                        ) as e:
                            logger.warning(
                                f"Invalid image data for URL: "
                                f"{image_url} "
                                f"(attempt {attempt + 1}/{max_retries}): {e}",
                            )
                            return None
                    else:
                        logger.warning(
                            f"Empty response body for image URL: "
                            f"{image_url} "
                            f"(attempt {attempt + 1}/{max_retries})",
                        )
                        return None
            except (
                requests.exceptions.RequestException,
                UnidentifiedImageError,
                OSError,
                ValueError,
            ) as e:
                if attempt == max_retries - 1:
                    logger.warning(
                        f"Failed to download image after "
                        f"{max_retries} attempts: {image_url} "
                        f"(error: {type(e).__name__}: {e})",
                    )
                    return None
                delay = base_delay * (2**attempt) + random.uniform(0, 0.1 * base_delay)
                logger.debug(
                    f"Retry {attempt + 1}/{max_retries}: {image_url}, "
                    f"sleeping {delay:.3f}s "
                    f"(error: {type(e).__name__}: {e})",
                )
                time.sleep(delay)

        return None

    def _load_image(self, image: Image.Image | dict | str | None) -> Image.Image | None:
        """Load image from various input types.

        Args:
            image: Image input - can be Image.Image, dict with "bytes" key,
                URL string, local file path, or None

        Returns:
            PIL Image in RGB mode if successful, None otherwise

        Raises:
            UnsupportedImageTypeError: If image type is not supported

        """
        if image is None:
            return None

        if isinstance(image, Image.Image):
            pass
        elif isinstance(image, dict) and "bytes" in image:
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, str):
            if image.startswith(("http://", "https://")):
                image = self._download_image_with_retry(
                    image_url=image,
                    max_retries=3,
                    base_delay=1.0,
                    max_size_bytes=self.settings.max_image_bytes,
                )
            else:
                try:
                    image = Image.open(image)
                except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                    logger.exception("Failed to open image file: %s - %s", image, type(e).__name__)
                    image = None
        else:
            raise UnsupportedImageTypeError(type(image))

        if image is not None and image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def index(
        self,
        alpha_values: list[float],
        rebuild: bool,
        batch_size: int = 100,
        sample_size: int | None = None,
    ) -> dict[float, VectorStore]:
        """Create vector store indexes for multiple alpha values.

        Pre-computes embeddings for each item and creates separate collections
        for each alpha value.

        Args:
            alpha_values: List of alpha values to index
            rebuild: If True, deletes existing collections before indexing (expensive).
                If False, reuses existing collections (default for user scripts).
            batch_size: Batch size for processing
            sample_size: Optional sample size (overrides settings)

        Returns:
            Dictionary mapping alpha values to VectorStore instances

        """
        dataset = self.dataset_loader.load()
        dataset_name = self.dataset_loader.get_dataset_name()
        model_id = self.embedding_model.get_model_id()
        environment = self.settings.environment

        if isinstance(dataset, DatasetDict):
            if "train" in dataset:
                dataset = dataset["train"]
                logger.info("Selected 'train' split from DatasetDict")
            else:
                first_split = next(iter(dataset.keys()))
                dataset = dataset[first_split]
                logger.info(
                    f"Selected '{first_split}' split from DatasetDict (no 'train' split available)",
                )

        if sample_size is not None:
            limit = sample_size
        elif environment == "dev":
            limit = self.settings.dev_sample_size
        else:
            limit = None

        dataset_len = len(dataset)
        if limit is not None:
            try:
                selected = dataset.select(range(min(limit, dataset_len)))
                if len(selected) > 0:
                    dataset = selected
                    dataset_len = len(dataset)
                    logger.info(
                        f"Successfully created sample: requested limit={limit}, "
                        f"selected length={dataset_len}, dataset type={type(dataset).__name__}",
                    )
            except (AttributeError, TypeError) as e:
                logger.warning(
                    f"Sampling failed, falling back to full dataset: "
                    f"exception={type(e).__name__}:{e}, "
                    f"dataset type={type(dataset).__name__}, dataset length={dataset_len}",
                )

        self.embedding_model.load_model()

        vectorstores: dict[float, VectorStore] = {}
        chroma_client = chromadb.PersistentClient(
            path=str(self.settings.chroma_db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        for alpha in alpha_values:
            collection_name = self.get_collection_name(
                dataset_name=dataset_name,
                model_id=model_id,
                alpha=alpha,
                environment=environment,
            )

            if rebuild:
                logger.info(f"Attempting to delete collection: {collection_name}")
                try:
                    chroma_client.delete_collection(name=collection_name)
                    logger.info(f"Successfully deleted collection: {collection_name}")
                except (ValueError, NotFoundError) as e:
                    logger.info(
                        f"Collection {collection_name} did not exist or deletion failed: {e}",
                    )

            vectorstore = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=EmbeddingModelWrapper(self.embedding_model),
            )

            for idx in tqdm(
                range(0, dataset_len, batch_size),
                desc=f"[Model: {model_id}] [Alpha: {alpha:.3f}]",
            ):
                batch = dataset[idx : idx + batch_size]

                if not isinstance(batch, dict):
                    batch = {col: batch[col] for col in batch.column_names}

                batch_size_actual = len(batch["text"])

                batch_documents = []
                batch_embeddings = []
                batch_metadatas = []
                batch_ids = []

                for i in range(batch_size_actual):
                    text = batch["text"][i]
                    image = batch["image"][i]

                    try:
                        image = self._load_image(image)

                        embedding = self.embedding_model.embed_multimodal(
                            image=image,
                            text=text,
                            alpha=alpha,
                        )

                        if not hasattr(embedding, "tolist"):
                            raise InvalidEmbeddingError(type(embedding))

                        doc_text = text if text is not None else f"[image-only sample {idx + i}]"
                        batch_documents.append(doc_text)
                        batch_embeddings.append(embedding.tolist())
                        batch_metadatas.append(
                            {
                                "alpha": alpha,
                                "index": idx + i,
                                "has_text": text is not None,
                                "has_image": image is not None,
                            },
                        )
                        batch_ids.append(f"{idx + i}")
                    except Exception as e:
                        logger.exception(
                            f"Failed to process item {idx + i} (alpha={alpha:.3f})",
                        )
                        raise RuntimeError(f"Embedding failed for item {idx + i}") from e

                vectorstore.add_texts(
                    texts=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )

            vectorstores[alpha] = vectorstore

        return vectorstores

    def create_vectorstore(
        self,
        dataset_name: str,
        model_id: str,
        alpha: float,
        environment: str = "dev",
    ) -> VectorStore:
        """Create VectorStore instance for existing collection.

        Args:
            dataset_name: Dataset identifier
            model_id: Model identifier
            alpha: Alpha value
            environment: Environment (dev/staging/prod)

        Returns:
            VectorStore instance

        """
        collection_name = self.get_collection_name(
            dataset_name=dataset_name,
            model_id=model_id,
            alpha=alpha,
            environment=environment,
        )

        chroma_client = chromadb.PersistentClient(
            path=str(self.settings.chroma_db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        try:
            chroma_client.get_collection(name=collection_name)
        except NotFoundError:
            raise CollectionNotFoundError(
                collection_name=collection_name,
                dataset_name=dataset_name,
                model_id=model_id,
                alpha=alpha,
                environment=environment,
            ) from None

        return Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=EmbeddingModelWrapper(self.embedding_model),
        )

    def get_embeddings(self) -> Embeddings:
        """Return embeddings wrapper instance.

        Returns:
            EmbeddingModelWrapper that can embed queries

        """
        return EmbeddingModelWrapper(self.embedding_model)

    def query_multimodal(
        self,
        vectorstore: VectorStore,
        image: Image.Image | str | None,
        text: str | None,
        alpha: float,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[tuple[Any, float]]:
        """Query vectorstore with multimodal embedding.

        This method performs similarity search using a multimodal query embedding
        that combines image and text according to the specified alpha value.
        This is essential for proper multimodal retrieval - using the standard
        similarity_search() method would only use text, ignoring the image component.

        Args:
            vectorstore: VectorStore instance to query (must be a Chroma instance)
            image: Query image (PIL Image, URL string, local path, or None)
            text: Query text string or None
            alpha: Alpha value for weighting (0.0 = text-only, 1.0 = image-only)
            k: Number of similar results to return
            score_threshold: Optional minimum similarity score threshold

        Returns:
            List of (Document, score) tuples sorted by similarity (higher is better)

        Raises:
            ValueError: If both image and text are None, or if alpha is invalid
            TypeError: If vectorstore is not a Chroma instance

        Example:
            >>> indexer = ChromaIndexer(dataset_loader, embedding_model, settings)
            >>> vectorstore = indexer.create_vectorstore("dataset", "model", 0.5, "dev")
            >>> results = indexer.query_multimodal(
            ...     vectorstore=vectorstore,
            ...     image=query_image,
            ...     text="red dress",
            ...     alpha=0.5,
            ...     k=5
            ... )
            >>> for doc, score in results:
            ...     print(f"Score: {score:.3f}, Text: {doc.page_content[:50]}")

        """
        if not isinstance(vectorstore, Chroma):
            raise TypeError(
                f"vectorstore must be a Chroma instance, got {type(vectorstore).__name__}"
            )

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in range [0.0, 1.0], got {alpha}")

        loaded_image = self._load_image(image)

        query_embedding = self.embedding_model.embed_multimodal(
            image=loaded_image,
            text=text,
            alpha=alpha,
        )

        query_embedding_list = query_embedding.tolist()

        chroma_collection = vectorstore._collection

        results = chroma_collection.query(
            query_embeddings=[query_embedding_list],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )

        documents = []
        if results and results["ids"] and len(results["ids"]) > 0:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            docs = results["documents"][0] if results["documents"] else [""] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

            for _doc_id, metadata, doc_text, distance in zip(
                ids,
                metadatas,
                docs,
                distances,
                strict=True,
            ):
                similarity_score = max(0.0, 1.0 - (distance**2 / 2.0))

                if score_threshold is not None and similarity_score < score_threshold:
                    continue

                from langchain_core.documents import Document

                doc = Document(page_content=doc_text, metadata=metadata)
                documents.append((doc, similarity_score))

        return documents

    def list_available_alphas(
        self,
        dataset_name: str,
        model_id: str,
        environment: str = "dev",
    ) -> list[float]:
        """List available alpha values for a dataset/model combination.

        Args:
            dataset_name: Dataset identifier
            model_id: Model identifier
            environment: Environment (dev/staging/prod)

        Returns:
            Sorted list of available alpha values

        """
        chroma_client = chromadb.PersistentClient(
            path=str(self.settings.chroma_db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        collections = chroma_client.list_collections()

        alphas = []
        prefix = self.get_collection_name(
            dataset_name=dataset_name,
            model_id=model_id,
            environment=environment,
        )

        for collection in collections:
            if collection.name.startswith(prefix):
                match = re.search(r"_a(\d+\.\d+)", collection.name)
                if match:
                    alphas.append(float(match.group(1)))

        return sorted(alphas)
