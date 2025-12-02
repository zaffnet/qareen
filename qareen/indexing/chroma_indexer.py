from __future__ import annotations

import contextlib
import logging
import math
import statistics
import time
from typing import TYPE_CHECKING, Any

from chromadb.errors import NotFoundError
from datasets import DatasetDict
from tqdm import tqdm

from qareen.indexing.base import VectorStoreIndexer
from qareen.indexing.embedding_model import EmbeddingModel
from qareen.models import Settings
from qareen.utils.chroma_client import close_chroma_client, create_chroma_client
from qareen.utils.image_utils import load_image
from qareen.utils.naming import get_collection_name

if TYPE_CHECKING:
    from qareen.dataset.base import DatasetLoader


logger = logging.getLogger(__name__)


def _estimate_batch_duration(batch_durations: list[float], lookback: int = 5) -> float:
    """Estimate batch duration using trimmed mean to exclude outliers.

    Uses median for small samples, trimmed mean (excluding top/bottom 25%) for larger samples.

    Args:
        batch_durations: Historical batch durations.
        lookback: Number of recent batches to consider.

    Returns:
        Estimated duration for next batch with 10% safety margin.
    """
    recent = batch_durations[-lookback:] if len(batch_durations) >= lookback else batch_durations
    if len(recent) == 1:
        return recent[0] * 1.1

    sorted_recent = sorted(recent)
    if len(sorted_recent) <= 3:
        median_duration = statistics.median(sorted_recent)
        return median_duration * 1.1

    trim_count = max(1, len(sorted_recent) // 4)
    trimmed = sorted_recent[trim_count:-trim_count] if trim_count > 0 else sorted_recent
    trimmed_mean = statistics.mean(trimmed) if trimmed else statistics.median(sorted_recent)
    return trimmed_mean * 1.1


class ChromaIndexer(VectorStoreIndexer):
    def __init__(
        self,
        dataset_loader: DatasetLoader,
        embedding_model: EmbeddingModel,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.settings.ensure_directories()
        self.dataset_loader = dataset_loader
        self.embedding_model = embedding_model
        self._chroma_client = None

    def _get_chroma_client(self):
        if self._chroma_client is None:
            self._chroma_client = create_chroma_client(self.settings.chroma_db_dir)
        return self._chroma_client

    def close(self) -> None:
        close_chroma_client(self._chroma_client)
        self._chroma_client = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def __enter__(self) -> ChromaIndexer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def index(
        self,
        alpha_values: list[float],
        *,
        rebuild: bool,
        batch_size: int = 100,
        sample_size: int | None = None,
        environment: str | None = None,
        timeout: float | None = None,
        log_progress_percent: float = 10.0,
    ) -> dict[float, Any]:
        """Index dataset with embedding model.

        Timeout enforcement operates at batch granularity: before starting each batch,
        the remaining timeout budget is checked. WARNING: The timeout is NOT enforced
        during batch processing. A single slow batch can exceed the entire timeout budget.
        The timeout is best-effort only.

        Args:
            timeout: Override timeout from Settings. If None, uses self.settings.timeout.
                When explicitly provided, must be positive (gt=0).
        """
        if timeout is None:
            timeout = self.settings.timeout
        elif timeout <= 0:
            raise ValueError(f"invalid timeout: {timeout}; must be positive (gt=0)")
        if not 0 < log_progress_percent <= 100:
            raise ValueError(
                f"invalid log_progress_percent: {log_progress_percent}; must be in range (0, 100]"
            )
        start_time = time.time()
        dataset = self.dataset_loader.load()
        if isinstance(dataset, DatasetDict):
            dataset = dataset.get("train", next(iter(dataset.values())))

        env = environment or self.settings.environment
        limit = (
            sample_size
            if sample_size is not None
            else (self.settings.dev_sample_size if env == "dev" else None)
        )
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        self.embedding_model.load_model()
        client = self._get_chroma_client()
        dataset_name = self.dataset_loader.get_dataset_name()
        model_id = self.embedding_model.get_model_id()
        vectorstores: dict[float, Any] = {}
        batch_durations: list[float] = []

        for alpha in alpha_values:
            name = get_collection_name(dataset_name, model_id, alpha, env)
            if rebuild:
                with contextlib.suppress(ValueError, NotFoundError):
                    client.delete_collection(name=name)

            collection = client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
            dataset_len = len(dataset)
            total_batches = math.ceil(dataset_len / batch_size)
            log_interval = max(1, int(total_batches * log_progress_percent / 100))

            for batch_idx, idx in enumerate(
                tqdm(
                    range(0, dataset_len, batch_size),
                    desc=f"[Model: {model_id}] [Alpha: {alpha:.3f}]",
                ),
                start=1,
            ):
                if timeout:
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        logger.error(
                            "Timeout exhausted before batch %d (elapsed: %.2fs, timeout: %.2fs)",
                            batch_idx,
                            elapsed,
                            timeout,
                        )
                        raise TimeoutError(
                            f"timeout exhausted before batch {batch_idx}; "
                            f"elapsed: {elapsed:.2f}s, timeout: {timeout:.2f}s"
                        )
                    if batch_durations:
                        estimated_duration = _estimate_batch_duration(batch_durations)
                        safety_margin = max(0.5, estimated_duration * 0.05)
                        if remaining <= estimated_duration + safety_margin:
                            logger.error(
                                "Insufficient time for estimated batch %d; "
                                "remaining: %.2fs, estimated: %.2fs, safety_margin: %.2fs",
                                batch_idx,
                                remaining,
                                estimated_duration,
                                safety_margin,
                            )
                            raise TimeoutError(
                                f"insufficient time for estimated batch {batch_idx}; "
                                f"remaining: {remaining:.2f}s, "
                                f"estimated: {estimated_duration:.2f}s, "
                                f"timeout: {timeout:.2f}s, elapsed: {elapsed:.2f}s"
                            )

                batch_start_time = time.time()
                batch = dataset[idx : idx + batch_size]
                batch_dict = (
                    batch
                    if isinstance(batch, dict)
                    else {col: batch[col] for col in batch.column_names}
                )

                docs, embeddings, metadatas, ids = [], [], [], []
                for i, (text, image) in enumerate(
                    zip(batch_dict["text"], batch_dict["image"], strict=True)
                ):
                    img = load_image(image)
                    emb = self.embedding_model.embed_multimodal(image=img, text=text, alpha=alpha)
                    if not hasattr(emb, "tolist"):
                        raise TypeError(f"Embedding must have tolist() method, got {type(emb)}")

                    docs.append(text or f"[image-only sample {idx + i}]")
                    embeddings.append(emb.tolist())
                    metadatas.append(
                        {
                            "alpha": alpha,
                            "index": idx + i,
                            "has_text": text is not None,
                            "has_image": img is not None,
                        }
                    )
                    ids.append(f"{idx + i}")

                if timeout:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        logger.error(
                            "Timeout exceeded during batch processing (elapsed: %.2fs)",
                            elapsed,
                        )
                        raise TimeoutError(f"Indexing timed out after {timeout} seconds")

                collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)
                if batch_idx == 1 or batch_idx % log_interval == 0:
                    processed_count = min(idx + batch_size, dataset_len)
                    logger.info(
                        "Progress: processed %d/%d items for alpha=%.3f",
                        processed_count,
                        dataset_len,
                        alpha,
                    )
                batch_durations.append(time.time() - batch_start_time)

            vectorstores[alpha] = collection

        return vectorstores
