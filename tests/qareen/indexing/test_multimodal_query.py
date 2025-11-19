"""Unit tests for multimodal query functionality in ChromaIndexer."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from datasets import Dataset
from PIL import Image

from qareen.config.settings import Settings
from qareen.dataset.base import DatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.models import EmbeddingModel


class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing multimodal queries."""

    def __init__(self, embedding_dim: int = 128, *, deterministic: bool = True) -> None:
        """Initialize mock model.

        Args:
            embedding_dim: Dimension of embeddings
            deterministic: If True, embeddings are deterministic based on inputs

        """
        self._embedding_dim = embedding_dim
        self.deterministic = deterministic
        self._model_loaded = False
        self.embed_text_calls: list[str | None] = []
        self.embed_image_calls: list[Any] = []
        self.embed_multimodal_calls: list[tuple[Any, str | None, float]] = []

    def load_model(self) -> None:
        """Load model (mock)."""
        self._model_loaded = True

    def embed_text(self, text: str | None) -> np.ndarray | None:
        """Generate text embedding.

        Args:
            text: Input text

        Returns:
            Embedding or None if text is None

        """
        self.embed_text_calls.append(text)
        if text is None:
            return None

        if self.deterministic:
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            embedding = rng.randn(self._embedding_dim).astype(np.float32)
        else:
            embedding = np.random.randn(self._embedding_dim).astype(np.float32)

        return self.normalize_l2(embedding)

    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        """Generate image embedding.

        Args:
            image: Input image

        Returns:
            Embedding or None if image is None

        """
        self.embed_image_calls.append(image)
        if image is None:
            return None

        if self.deterministic and isinstance(image, Image.Image):
            image_array = np.array(image)
            seed = int(image_array.sum()) % (2**31)
            rng = np.random.RandomState(seed)
            embedding = rng.randn(self._embedding_dim).astype(np.float32)
        else:
            embedding = np.random.randn(self._embedding_dim).astype(np.float32)

        return self.normalize_l2(embedding)

    def embed_multimodal(
        self,
        image: Image.Image | str | Path | None,
        text: str | None,
        alpha: float,
    ) -> np.ndarray:
        """Generate multimodal embedding.

        Args:
            image: Input image
            text: Input text
            alpha: Weight for image embedding

        Returns:
            Combined embedding

        """
        self.embed_multimodal_calls.append((image, text, alpha))
        image_embedding = self.embed_image(image)
        text_embedding = self.embed_text(text)

        if image_embedding is None and text_embedding is None:
            raise ValueError("At least one modality must be present")

        if image_embedding is None:
            return text_embedding

        if text_embedding is None:
            return image_embedding

        combined = alpha * image_embedding + (1 - alpha) * text_embedding
        return self.normalize_l2(combined)

    def get_model_id(self) -> str:
        """Return model identifier.

        Returns:
            Model ID string

        """
        return "mock_model"

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension.

        Returns:
            Embedding dimension

        """
        return self._embedding_dim


class SimpleDatasetLoader(DatasetLoader):
    """Simple dataset loader for testing."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        """Initialize with sample data.

        Args:
            samples: List of sample dictionaries with 'text' and 'image' keys

        """
        self.samples = samples
        self._dataset: Dataset | None = None

    def load(self) -> Dataset:
        """Load dataset.

        Returns:
            Dataset instance

        """
        if self._dataset is None:
            self._dataset = Dataset.from_list(self.samples)
        return self._dataset

    def validate_schema(self) -> None:
        """Validate dataset schema."""
        pass

    def get_dataset_name(self) -> str:
        """Return dataset name.

        Returns:
            Dataset name

        """
        return "test_dataset"

    def get_dataset_info(self) -> dict[str, object]:
        """Return dataset info.

        Returns:
            Dataset info dictionary

        """
        return {"size": len(self.samples)}


def test_query_multimodal_with_text_only() -> None:
    """Test multimodal query with text-only data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = MockEmbeddingModel(embedding_dim=128)

        samples = [
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

        vectorstores = indexer.index(alpha_values=[0.5], rebuild=True, batch_size=10)
        vectorstore = vectorstores[0.5]

        results = indexer.query_multimodal(
            vectorstore=vectorstore,
            image=None,
            text="apple",
            alpha=0.5,
            k=2,
        )

        assert len(results) <= 2
        assert all(isinstance(doc, tuple) for doc in results)
        assert all(len(doc) == 2 for doc in results)


def test_query_multimodal_with_image_only() -> None:
    """Test multimodal query with image-only data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = MockEmbeddingModel(embedding_dim=128)

        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color="green")
        img3 = Image.new("RGB", (100, 100), color="blue")

        samples = [
            {"text": None, "image": img1},
            {"text": None, "image": img2},
            {"text": None, "image": img3},
        ]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[1.0], rebuild=True, batch_size=10)
        vectorstore = vectorstores[1.0]

        query_image = Image.new("RGB", (100, 100), color="red")
        results = indexer.query_multimodal(
            vectorstore=vectorstore,
            image=query_image,
            text=None,
            alpha=1.0,
            k=2,
        )

        assert len(results) <= 2
        assert all(isinstance(doc, tuple) for doc in results)


def test_query_multimodal_with_both_modalities() -> None:
    """Test multimodal query with both text and image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = MockEmbeddingModel(embedding_dim=128)

        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color="green")

        samples = [
            {"text": "red apple", "image": img1},
            {"text": "green banana", "image": img2},
        ]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], rebuild=True, batch_size=10)
        vectorstore = vectorstores[0.5]

        query_image = Image.new("RGB", (100, 100), color="red")
        results = indexer.query_multimodal(
            vectorstore=vectorstore,
            image=query_image,
            text="red apple",
            alpha=0.5,
            k=2,
        )

        assert len(results) <= 2
        assert all(isinstance(doc, tuple) for doc in results)
        for _doc, score in results:
            assert 0.0 <= score <= 1.0


def test_query_multimodal_different_alphas_different_results() -> None:
    """Test that different alpha values produce different query results.

    This is the CRITICAL test that would have caught the original bug.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = MockEmbeddingModel(embedding_dim=128, deterministic=True)

        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(0, 255, 0))
        img3 = Image.new("RGB", (100, 100), color=(0, 0, 255))

        samples = [
            {"text": "text_alpha", "image": img1},
            {"text": "text_beta", "image": img2},
            {"text": "text_gamma", "image": img3},
        ]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        alpha_values = [0.0, 0.5, 1.0]
        vectorstores = indexer.index(alpha_values=alpha_values, rebuild=True, batch_size=10)

        query_image = Image.new("RGB", (100, 100), color=(255, 0, 0))
        query_text = "text_alpha"

        all_results = {}
        for alpha in alpha_values:
            vectorstore = vectorstores[alpha]
            results = indexer.query_multimodal(
                vectorstore=vectorstore,
                image=query_image,
                text=query_text,
                alpha=alpha,
                k=3,
            )
            all_results[alpha] = results

        results_0_0 = all_results[0.0]
        results_0_5 = all_results[0.5]
        results_1_0 = all_results[1.0]

        assert len(results_0_0) > 0
        assert len(results_0_5) > 0
        assert len(results_1_0) > 0

        score_0_0 = results_0_0[0][1] if results_0_0 else None
        score_0_5 = results_0_5[0][1] if results_0_5 else None
        score_1_0 = results_1_0[0][1] if results_1_0 else None

        assert score_0_0 is not None
        assert score_0_5 is not None
        assert score_1_0 is not None

        scores_differ = (
            not np.isclose(score_0_0, score_1_0, rtol=1e-3)
            or not np.isclose(score_0_0, score_0_5, rtol=1e-3)
            or not np.isclose(score_0_5, score_1_0, rtol=1e-3)
        )
        assert scores_differ, (
            f"Scores should differ for different alphas, but got: "
            f"alpha=0.0: {score_0_0:.4f}, alpha=0.5: {score_0_5:.4f}, alpha=1.0: {score_1_0:.4f}"
        )


def test_query_multimodal_validates_alpha() -> None:
    """Test that query_multimodal validates alpha range."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = MockEmbeddingModel(embedding_dim=128)

        samples = [{"text": "sample", "image": None}]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], rebuild=True, batch_size=10)
        vectorstore = vectorstores[0.5]

        with pytest.raises(ValueError) as exc:
            indexer.query_multimodal(
                vectorstore=vectorstore,
                image=None,
                text="query",
                alpha=1.5,
                k=1,
            )
        assert "alpha must be in range" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            indexer.query_multimodal(
                vectorstore=vectorstore,
                image=None,
                text="query",
                alpha=-0.1,
                k=1,
            )
        assert "alpha must be in range" in str(exc.value)


def test_query_multimodal_with_score_threshold() -> None:
    """Test query_multimodal with score threshold filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = MockEmbeddingModel(embedding_dim=128)

        samples = [
            {"text": "apple", "image": None},
            {"text": "banana", "image": None},
            {"text": "cherry", "image": None},
        ]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], rebuild=True, batch_size=10)
        vectorstore = vectorstores[0.5]

        results_no_threshold = indexer.query_multimodal(
            vectorstore=vectorstore,
            image=None,
            text="apple",
            alpha=0.5,
            k=3,
        )

        results_with_threshold = indexer.query_multimodal(
            vectorstore=vectorstore,
            image=None,
            text="apple",
            alpha=0.5,
            k=3,
            score_threshold=0.9,
        )

        assert len(results_with_threshold) <= len(results_no_threshold)
        for _doc, score in results_with_threshold:
            assert score >= 0.9


def test_query_multimodal_calls_embed_multimodal() -> None:
    """Test that query_multimodal actually calls embed_multimodal on the model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = MockEmbeddingModel(embedding_dim=128)

        samples = [{"text": "sample", "image": None}]
        loader = SimpleDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], rebuild=True, batch_size=10)
        vectorstore = vectorstores[0.5]

        initial_call_count = len(model.embed_multimodal_calls)

        indexer.query_multimodal(
            vectorstore=vectorstore,
            image=None,
            text="query text",
            alpha=0.5,
            k=1,
        )

        assert len(model.embed_multimodal_calls) > initial_call_count
        last_call = model.embed_multimodal_calls[-1]
        assert last_call[1] == "query text"
        assert last_call[2] == 0.5
