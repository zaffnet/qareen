"""ChromaDB vector store indexer implementation."""

from __future__ import annotations

import logging
import random
import re
import time
from io import BytesIO
from typing import cast

import chromadb
import requests
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from qareen.config.settings import Settings
from qareen.dataset.base import DatasetLoader
from qareen.indexing.base import VectorStoreIndexer
from qareen.indexing.exceptions import (
    CollectionNotFoundError,
    InvalidEmbeddingError,
    UnsupportedImageTypeError,
)
from qareen.indexing.models import EmbeddingModel

logger = logging.getLogger(__name__)


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
                        f"Model: {model_id}, Text: {text[:100] if text else 'None'}..."
                    )
                embedding_list = cast(list[float], embedding.tolist())
                if len(embedding_list) != self._embedding_dim:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                        f"got {len(embedding_list)}"
                    )
                embeddings.append(embedding_list)
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
        """
        embedding = self.embedding_model.embed_text(text)
        if embedding is None:
            model_id = self.embedding_model.get_model_id()
            raise ValueError(
                f"Embedding returned None for provided text. "
                f"Model: {model_id}, Text: {text[:100] if text else 'None'}..."
            )
        return cast(list[float], embedding.tolist())


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
                response = requests.get(image_url, timeout=30, stream=True)
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "").lower()
                if not content_type.startswith("image/"):
                    logger.warning(
                        f"Invalid Content-Type '{content_type}' "
                        f"for image URL: {image_url} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    response.close()
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
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            response.close()
                            if attempt == max_retries - 1:
                                return None
                            delay = base_delay * (2**attempt) + random.uniform(0, 0.1 * base_delay)
                            time.sleep(delay)
                            continue
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Invalid Content-Length header "
                            f"'{content_length}' for image URL: "
                            f"{image_url} "
                            f"(attempt {attempt + 1}/{max_retries}): "
                            f"{e}"
                        )
                        response.close()
                        if attempt == max_retries - 1:
                            return None
                        delay = base_delay * (2**attempt) + random.uniform(0, 0.1 * base_delay)
                        time.sleep(delay)
                        continue

                content = bytearray()
                try:
                    for chunk in response.iter_content(chunk_size=8192):
                        if len(content) + len(chunk) > max_size_bytes:
                            logger.warning(
                                f"Image size exceeds max "
                                f"{max_size_bytes} bytes during "
                                f"download for image URL: {image_url} "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            response.close()
                            return None
                        content.extend(chunk)
                    else:
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
                                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                                )
                                return None
                        else:
                            logger.warning(
                                f"Empty response body for image URL: "
                                f"{image_url} "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            return None
                finally:
                    response.close()
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
                        f"(error: {type(e).__name__}: {e})"
                    )
                    return None
                else:
                    delay = base_delay * (2**attempt) + random.uniform(0, 0.1 * base_delay)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_retries}: {image_url}, "
                        f"sleeping {delay:.2f}s "
                        f"(error: {type(e).__name__}: {e})"
                    )
                    time.sleep(delay)

        return None

    def index(
        self,
        alpha_values: list[float],
        batch_size: int = 100,
        sample_size: int | None = None,
    ) -> dict[float, VectorStore]:
        """Create vector store indexes for multiple alpha values.

        Pre-computes embeddings for each item and creates separate collections
        for each alpha value.

        Args:
            alpha_values: List of alpha values to index
            batch_size: Batch size for processing
            sample_size: Optional sample size (overrides settings)

        Returns:
            Dictionary mapping alpha values to VectorStore instances
        """
        dataset = self.dataset_loader.load()
        dataset_name = self.dataset_loader.get_dataset_name()
        model_id = self.embedding_model.get_model_id()
        environment = self.settings.environment

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
                        f"selected length={dataset_len}, dataset type={type(dataset).__name__}"
                    )
            except (AttributeError, TypeError) as e:
                logger.warning(
                    f"Sampling failed, falling back to full dataset: "
                    f"exception={type(e).__name__}:{e}, "
                    f"dataset type={type(dataset).__name__}, dataset length={dataset_len}"
                )

        self.embedding_model.load_model()

        vectorstores: dict[float, VectorStore] = {}
        chroma_client = chromadb.PersistentClient(path=str(self.settings.chroma_db_dir))

        for alpha in alpha_values:
            collection_name = self.get_collection_name(
                dataset_name=dataset_name,
                model_id=model_id,
                alpha=alpha,
                environment=environment,
            )

            logger.info(f"Attempting to delete collection: {collection_name}")
            try:
                chroma_client.delete_collection(name=collection_name)
                logger.info(f"Successfully deleted collection: {collection_name}")
            except (ValueError, NotFoundError) as e:
                logger.info(f"Collection {collection_name} did not exist or deletion failed: {e}")

            vectorstore = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=EmbeddingModelWrapper(self.embedding_model),
            )

            for idx in tqdm(
                range(0, dataset_len, batch_size),
                desc=f"[Model: {model_id}] [Alpha: {alpha:.2f}]",
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
                        if image is not None:
                            if isinstance(image, Image.Image):
                                pass
                            elif isinstance(image, dict) and "bytes" in image:
                                image = Image.open(BytesIO(image["bytes"]))
                            elif isinstance(image, str):
                                if image.startswith("http://") or image.startswith("https://"):
                                    image = self._download_image_with_retry(
                                        image_url=image,
                                        max_retries=3,
                                        base_delay=1.0,
                                        max_size_bytes=self.settings.max_image_bytes,
                                    )
                                else:
                                    try:
                                        image = Image.open(image)
                                    except FileNotFoundError:
                                        logger.exception(f"Image file not found: {image}")
                                        image = None
                                    except UnidentifiedImageError:
                                        logger.exception(f"Cannot identify image file: {image}")
                                        image = None
                                    except Exception:
                                        logger.exception(f"Failed to open image file: {image}")
                                        image = None
                            else:
                                raise UnsupportedImageTypeError(type(image))

                            if image is not None and image.mode != "RGB":
                                image = image.convert("RGB")

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
                            }
                        )
                        batch_ids.append(f"{idx + i}")
                    except Exception as e:
                        logger.exception(
                            f"Failed to process item {idx + i} (alpha={alpha:.2f})",
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

        chroma_client = chromadb.PersistentClient(path=str(self.settings.chroma_db_dir))

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
        chroma_client = chromadb.PersistentClient(path=str(self.settings.chroma_db_dir))

        collections = chroma_client.list_collections()

        alphas = []
        prefix = self.get_collection_name(
            dataset_name=dataset_name,
            model_id=model_id,
            environment=environment,
        )

        for collection in collections:
            if collection.name.startswith(prefix):
                match = re.search(r"alpha(\d+(?:\.\d+)?)", collection.name)
                if match:
                    alphas.append(float(match.group(1)))

        return sorted(alphas)
