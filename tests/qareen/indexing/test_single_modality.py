"""Tests for single-modality embedding and indexing support."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from qareen.config.settings import Settings
from qareen.dataset.base import DatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.models import EmbeddingModel

MISSING_MODALITY_ERROR = "At least one modality must be present"


class SingleModalityEmbeddingModel(EmbeddingModel):
    """Mock embedding model supporting single-modality inputs."""

    def __init__(self, embedding_dim: int = 128) -> None:
        """Initialize mock model.

        Args:
            embedding_dim: Embedding dimension
        """
        self._embedding_dim = embedding_dim
        self.model_loaded = False

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension.

        Returns:
            Embedding dimension
        """
        return self._embedding_dim

    def load_model(self) -> None:
        """Mark model as loaded."""
        self.model_loaded = True

    def embed_text(self, text: str | None) -> np.ndarray | None:
        """Generate deterministic text embedding or None if text is None.

        Args:
            text: Input text or None

        Returns:
            L2-normalized embedding vector or None
        """
        if text is None:
            return None
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        return self.normalize_l2(embedding)

    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        """Generate deterministic image embedding or None if image is None.

        Args:
            image: Input image or None

        Returns:
            L2-normalized embedding vector or None
        """
        if image is None:
            return None
        image_hash = hash(str(image.tobytes() if isinstance(image, Image.Image) else image))
        np.random.seed(image_hash % 2**32)
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        return self.normalize_l2(embedding)

    def embed_multimodal(
        self,
        image: Image.Image | str | Path | None,
        text: str | None,
        alpha: float,
    ) -> np.ndarray:
        """Generate combined embedding handling missing modalities.

        Args:
            image: Input image or None
            text: Input text or None
            alpha: Weight for image embedding

        Returns:
            L2-normalized combined embedding
        """
        image_emb = self.embed_image(image)
        text_emb = self.embed_text(text)

        if image_emb is None and text_emb is None:
            raise ValueError(MISSING_MODALITY_ERROR)

        if image_emb is None:
            assert text_emb is not None
            return text_emb

        if text_emb is None:
            return image_emb

        combined = alpha * image_emb + (1 - alpha) * text_emb
        return self.normalize_l2(combined)

    def get_model_id(self) -> str:
        """Return mock model ID.

        Returns:
            Model identifier
        """
        return "single_modality_mock"


class SingleModalityDatasetLoader(DatasetLoader):
    """Mock dataset loader with single-modality samples."""

    def __init__(self, samples: list[dict[str, object]]) -> None:
        """Initialize loader with samples.

        Args:
            samples: List of sample dicts with optional text/image
        """
        self.samples = samples

    def load(self) -> dict | MagicMock:
        """Return mock dataset with single-modality samples.

        Returns:
            Mock dataset
        """
        dataset = MagicMock()
        samples_len = len(self.samples)
        dataset.__len__ = lambda _: samples_len
        dataset.column_names = ["text", "image"]

        def getitem(*args: object) -> dict[str, object]:
            idx = args[-1] if args else slice(None)
            if isinstance(idx, slice):
                start = idx.start or 0
                stop = idx.stop or len(self.samples)
                selected = self.samples[start:stop]
                return {
                    "text": [s.get("text") for s in selected],
                    "image": [s.get("image") for s in selected],
                }
            assert isinstance(idx, int)
            return self.samples[idx]

        def select(indices: range) -> MagicMock:
            indices_list = list(indices)
            selected = MagicMock()
            indices_len = len(indices_list)
            selected.__len__ = lambda _: indices_len
            selected.column_names = ["text", "image"]

            def getitem_selected(*args: object) -> dict[str, object]:
                idx = args[-1] if args else slice(None)
                if isinstance(idx, slice):
                    start = idx.start or 0
                    stop = idx.stop or len(indices_list)
                    selected_indices = indices_list[start:stop]
                    return {
                        "text": [self.samples[i].get("text") for i in selected_indices],
                        "image": [self.samples[i].get("image") for i in selected_indices],
                    }
                assert isinstance(idx, int)
                return self.samples[indices_list[idx]]

            selected.__getitem__ = getitem_selected
            return selected

        dataset.__getitem__ = getitem
        dataset.select = select
        return dataset

    def get_dataset_name(self) -> str:
        """Return dataset name.

        Returns:
            Dataset identifier
        """
        return "single_modality_test"

    def validate_schema(self) -> None:
        """Validate schema."""
        pass

    def get_dataset_info(self) -> dict[str, object]:
        """Return dataset info."""
        return {"size": len(self.samples)}


def test_embedding_model_returns_none_for_missing_text() -> None:
    """Embedding model must return None when text is None."""
    model = SingleModalityEmbeddingModel(embedding_dim=128)
    result = model.embed_text(None)
    assert result is None


def test_embedding_model_returns_none_for_missing_image() -> None:
    """Embedding model must return None when image is None."""
    model = SingleModalityEmbeddingModel(embedding_dim=128)
    result = model.embed_image(None)
    assert result is None


def test_embedding_model_returns_text_only_embedding() -> None:
    """Embedding model must return text embedding when image is None."""
    model = SingleModalityEmbeddingModel(embedding_dim=128)
    result = model.embed_multimodal(image=None, text="sample text", alpha=0.5)
    assert result is not None
    assert len(result) == 128
    assert np.allclose(np.linalg.norm(result), 1.0, atol=1e-6)


def test_embedding_model_returns_image_only_embedding() -> None:
    """Embedding model must return image embedding when text is None."""
    model = SingleModalityEmbeddingModel(embedding_dim=128)
    img = Image.new("RGB", (224, 224), color="blue")
    result = model.embed_multimodal(image=img, text=None, alpha=0.5)
    assert result is not None
    assert len(result) == 128
    assert np.allclose(np.linalg.norm(result), 1.0, atol=1e-6)


def test_embedding_model_rejects_both_none() -> None:
    """Embedding model must reject when both modalities are None."""
    model = SingleModalityEmbeddingModel(embedding_dim=128)
    with pytest.raises(ValueError, match=r"(?i)at least one modality"):
        model.embed_multimodal(image=None, text=None, alpha=0.5)


def test_indexer_handles_text_only_samples() -> None:
    """Indexer must successfully index text-only samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = SingleModalityEmbeddingModel(embedding_dim=128)
        samples: list[dict[str, object]] = [
            {"text": "text only sample 1", "image": None},
            {"text": "text only sample 2", "image": None},
        ]
        loader = SingleModalityDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], rebuild=True, batch_size=10)

        assert len(vectorstores) == 1
        vectorstore = vectorstores[0.5]

        results = vectorstore.similarity_search("text only", k=2)
        assert len(results) == 2


def test_indexer_handles_image_only_samples() -> None:
    """Indexer must successfully index image-only samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = SingleModalityEmbeddingModel(embedding_dim=128)
        samples = [
            {"text": None, "image": Image.new("RGB", (224, 224), color="red")},
            {"text": None, "image": Image.new("RGB", (224, 224), color="blue")},
        ]
        loader = SingleModalityDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], rebuild=True, batch_size=10)

        assert len(vectorstores) == 1
        vectorstore = vectorstores[0.5]

        results = vectorstore.similarity_search("query", k=2)
        assert len(results) == 2


def test_indexer_handles_mixed_modality_samples() -> None:
    """Indexer must handle dataset with mixed single/dual modality samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = SingleModalityEmbeddingModel(embedding_dim=128)
        samples = [
            {"text": "both modalities", "image": Image.new("RGB", (224, 224), color="red")},
            {"text": "text only", "image": None},
            {"text": None, "image": Image.new("RGB", (224, 224), color="blue")},
        ]
        loader = SingleModalityDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], rebuild=True, batch_size=10)

        assert len(vectorstores) == 1
        vectorstore = vectorstores[0.5]

        results = vectorstore.similarity_search("query", k=3)
        assert len(results) == 3
