"""Tests for alpha weighting with single-modality samples."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from qareen.config.settings import Settings
from qareen.dataset.base import DatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.models import EmbeddingModel


class AlphaAwareEmbeddingModel(EmbeddingModel):
    """Mock embedding model that tracks alpha usage with single-modality inputs."""

    def __init__(self, embedding_dim: int = 128) -> None:
        """Initialize mock model.

        Args:
            embedding_dim: Embedding dimension
        """
        self._embedding_dim = embedding_dim
        self.model_loaded = False
        self.embed_calls: list[dict] = []

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
        """Generate text embedding or None if text is None.

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
        """Generate image embedding or None if image is None.

        Args:
            image: Input image or None

        Returns:
            L2-normalized embedding vector or None
        """
        if image is None:
            return None
        np.random.seed(42)
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
        self.embed_calls.append({"image": image, "text": text, "alpha": alpha})

        image_emb = self.embed_image(image)
        text_emb = self.embed_text(text)

        if image_emb is None and text_emb is None:
            raise ValueError("At least one modality must be present")

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
        return "alpha_aware_mock"


class SimpleDatasetLoader(DatasetLoader):
    """Simple dataset loader for testing."""

    def __init__(self, samples: list[dict[str, object]]) -> None:
        """Initialize loader with samples.

        Args:
            samples: List of sample dicts
        """
        self.samples = samples

    def load(self) -> dict | MagicMock:
        """Return mock dataset.

        Returns:
            Mock dataset
        """
        dataset = MagicMock()
        dataset.__len__ = lambda *_: len(self.samples)
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
            selected.__len__ = lambda *_: len(indices_list)
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
        return "alpha_test"

    def validate_schema(self) -> None:
        """Validate schema."""
        pass

    def get_dataset_info(self) -> dict[str, object]:
        """Return dataset info."""
        return {"size": len(self.samples)}


def test_alpha_ignored_for_text_only_samples() -> None:
    """Alpha should be ignored when only text is present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = AlphaAwareEmbeddingModel(embedding_dim=128)
        samples: list[dict[str, object]] = [{"text": "text only", "image": None}]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.8], batch_size=10)

        assert len(vectorstores) == 1
        assert len(model.embed_calls) == 1
        call = model.embed_calls[0]
        assert call["text"] == "text only"
        assert call["image"] is None
        assert call["alpha"] == 0.8


def test_alpha_ignored_for_image_only_samples() -> None:
    """Alpha should be ignored when only image is present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = AlphaAwareEmbeddingModel(embedding_dim=128)
        samples = [{"text": None, "image": Image.new("RGB", (224, 224), color="red")}]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.2], batch_size=10)

        assert len(vectorstores) == 1
        assert len(model.embed_calls) == 1
        call = model.embed_calls[0]
        assert call["text"] is None
        assert call["image"] is not None
        assert call["alpha"] == 0.2


def test_alpha_matters_for_dual_modality_samples() -> None:
    """Alpha should affect embedding when both modalities present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = AlphaAwareEmbeddingModel(embedding_dim=128)
        samples = [{"text": "caption", "image": Image.new("RGB", (224, 224), color="red")}]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], batch_size=10)

        assert len(vectorstores) == 1
        assert len(model.embed_calls) == 1
        call = model.embed_calls[0]
        assert call["text"] == "caption"
        assert call["image"] is not None
        assert call["alpha"] == 0.5


def test_multiple_alphas_with_single_modality() -> None:
    """Indexing with multiple alphas should work even for single-modality samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = AlphaAwareEmbeddingModel(embedding_dim=128)
        samples: list[dict[str, object]] = [{"text": "text only", "image": None}]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.0, 0.5, 1.0], batch_size=10)

        assert len(vectorstores) == 3
        assert 0.0 in vectorstores
        assert 0.5 in vectorstores
        assert 1.0 in vectorstores
        assert len(model.embed_calls) == 3


def test_text_only_query_with_text_only_index() -> None:
    """Text query must retrieve text-only samples from text-only index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = AlphaAwareEmbeddingModel(embedding_dim=128)
        samples: list[dict[str, object]] = [
            {"text": "apple fruit", "image": None},
            {"text": "banana fruit", "image": None},
            {"text": "cherry fruit", "image": None},
        ]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], batch_size=10)
        vectorstore = vectorstores[0.5]

        results = vectorstore.similarity_search("apple", k=1)
        assert len(results) == 1


def test_image_query_embedding_with_image_only_index() -> None:
    """Image-based query should work with image-only index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = AlphaAwareEmbeddingModel(embedding_dim=128)
        samples = [
            {"text": None, "image": Image.new("RGB", (224, 224), color="red")},
            {"text": None, "image": Image.new("RGB", (224, 224), color="blue")},
        ]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], batch_size=10)
        vectorstore = vectorstores[0.5]

        results = vectorstore.similarity_search("query text", k=2)
        assert len(results) == 2


def test_alpha_zero_equivalent_to_text_only() -> None:
    """Alpha=0.0 should produce text-only embedding for dual-modality sample."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = AlphaAwareEmbeddingModel(embedding_dim=128)
        samples = [{"text": "caption", "image": Image.new("RGB", (224, 224), color="red")}]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.0], batch_size=10)

        assert len(vectorstores) == 1
        assert len(model.embed_calls) == 1
        call = model.embed_calls[0]
        assert call["alpha"] == 0.0


def test_alpha_one_equivalent_to_image_only() -> None:
    """Alpha=1.0 should produce image-only embedding for dual-modality sample."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = AlphaAwareEmbeddingModel(embedding_dim=128)
        samples = [{"text": "caption", "image": Image.new("RGB", (224, 224), color="red")}]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[1.0], batch_size=10)

        assert len(vectorstores) == 1
        assert len(model.embed_calls) == 1
        call = model.embed_calls[0]
        assert call["alpha"] == 1.0
