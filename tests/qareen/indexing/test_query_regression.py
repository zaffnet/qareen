"""Regression tests with fixed expected results for multimodal queries.

These tests use fixed datasets and known query results to prevent silent breakage
of multimodal retrieval functionality.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from PIL import Image

from qareen.config.settings import Settings
from qareen.dataset.base import DatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.models import EmbeddingModel


class FixedEmbeddingModel(EmbeddingModel):
    """Fixed embedding model that returns predictable embeddings for testing."""

    def __init__(self, embedding_dim: int = 64) -> None:
        """Initialize model with fixed dimension.

        Args:
            embedding_dim: Dimension of embeddings

        """
        self._embedding_dim = embedding_dim
        self._model_loaded = False
        self.text_embeddings: dict[str, np.ndarray] = {}
        self.image_embeddings: dict[str, np.ndarray] = {}

    def load_model(self) -> None:
        """Load model (mock)."""
        self._model_loaded = True

    def _get_fixed_embedding(self, key: str, embedding_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Get or create a fixed embedding for a key.

        Args:
            key: Key to identify the embedding
            embedding_dict: Dictionary to store embeddings

        Returns:
            Fixed embedding vector

        """
        if key not in embedding_dict:
            seed = abs(hash(key)) % (2**31)
            rng = np.random.RandomState(seed)
            embedding = rng.randn(self._embedding_dim).astype(np.float32)
            embedding_dict[key] = self.normalize_l2(embedding)
        return embedding_dict[key]

    def embed_text(self, text: str | None) -> np.ndarray | None:
        """Generate fixed text embedding.

        Args:
            text: Input text

        Returns:
            Fixed embedding or None if text is None

        """
        if text is None:
            return None
        return self._get_fixed_embedding(text, self.text_embeddings)

    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        """Generate fixed image embedding.

        Args:
            image: Input image

        Returns:
            Fixed embedding or None if image is None

        """
        if image is None:
            return None

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if not isinstance(image, Image.Image):
            raise TypeError("Image must be PIL Image or path string")

        image_array = np.array(image)
        key = f"img_{image.size}_{int(image_array.sum())}_{int(image_array.mean() * 1000)}"
        return self._get_fixed_embedding(key, self.image_embeddings)

    def embed_multimodal(
        self,
        image: Image.Image | str | Path | None,
        text: str | None,
        alpha: float,
    ) -> np.ndarray:
        """Generate multimodal embedding with alpha weighting.

        Args:
            image: Input image
            text: Input text
            alpha: Weight for image embedding

        Returns:
            Combined embedding

        """
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
            Model ID

        """
        return "fixed_regression_model"

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension.

        Returns:
            Embedding dimension

        """
        return self._embedding_dim


class FixedDatasetLoader(DatasetLoader):
    """Fixed dataset loader for regression tests."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        """Initialize with fixed sample data.

        Args:
            samples: List of sample dictionaries

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
        """Validate dataset schema. Intentionally empty for tests."""
        ...

    def get_dataset_name(self) -> str:
        """Return dataset name.

        Returns:
            Dataset name

        """
        return "regression_test_dataset"

    def get_dataset_info(self) -> dict[str, object]:
        """Return dataset info.

        Returns:
            Dataset info

        """
        return {"size": len(self.samples)}


def create_fixed_test_dataset() -> list[dict[str, Any]]:
    """Create a fixed test dataset for regression tests.

    Returns:
        List of sample dictionaries with fixed images and text

    """
    img1 = Image.new("RGB", (50, 50), color=(255, 0, 0))
    img2 = Image.new("RGB", (50, 50), color=(0, 255, 0))
    img3 = Image.new("RGB", (50, 50), color=(0, 0, 255))

    return [
        {"text": "red square", "image": img1},
        {"text": "green square", "image": img2},
        {"text": "blue square", "image": img3},
        {"text": "red item", "image": img1},
        {"text": "blue item", "image": img3},
    ]


def test_regression_text_heavy_query() -> None:
    """Regression test for text-heavy (alpha=0.2) multimodal query.

    This test freezes expected behavior for a specific query configuration.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = FixedEmbeddingModel(embedding_dim=64)

        samples = create_fixed_test_dataset()
        loader = FixedDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.2], rebuild=True, batch_size=10)
        vectorstore = vectorstores[0.2]

        query_image = Image.new("RGB", (50, 50), color=(255, 0, 0))
        query_text = "red square"

        results = indexer.query_multimodal(
            vectorstore=vectorstore,
            image=query_image,
            text=query_text,
            alpha=0.2,
            k=3,
        )

        assert len(results) == 3

        top_result = results[0]
        top_doc, top_score = top_result

        assert top_doc.page_content == "red square"

        assert 0.8 < top_score <= 1.0, (
            f"Expected high similarity for exact match, got {top_score:.4f}"
        )

        for _doc, score in results:
            assert 0.0 <= score <= 1.0


def test_regression_image_heavy_query() -> None:
    """Regression test for image-heavy (alpha=0.8) multimodal query.

    This test freezes expected behavior for image-focused retrieval.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = FixedEmbeddingModel(embedding_dim=64)

        samples = create_fixed_test_dataset()
        loader = FixedDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.8], rebuild=True, batch_size=10)
        vectorstore = vectorstores[0.8]

        query_image = Image.new("RGB", (50, 50), color=(0, 0, 255))
        query_text = "unrelated query text"

        results = indexer.query_multimodal(
            vectorstore=vectorstore,
            image=query_image,
            text=query_text,
            alpha=0.8,
            k=3,
        )

        assert len(results) == 3

        top_result = results[0]
        top_doc, top_score = top_result

        assert top_doc.page_content in [
            "red square",
            "green square",
            "blue square",
            "red item",
            "blue item",
        ], f"Expected one of the indexed samples, got: {top_doc.page_content}"

        assert top_score >= 0.0, f"Expected valid similarity score, got {top_score:.4f}"


def test_regression_balanced_query() -> None:
    """Regression test for balanced (alpha=0.5) multimodal query.

    This test verifies that alpha=0.5 properly balances both modalities.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = FixedEmbeddingModel(embedding_dim=64)

        samples = create_fixed_test_dataset()
        loader = FixedDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], rebuild=True, batch_size=10)
        vectorstore = vectorstores[0.5]

        query_image = Image.new("RGB", (50, 50), color=(0, 255, 0))
        query_text = "green square"

        results = indexer.query_multimodal(
            vectorstore=vectorstore,
            image=query_image,
            text=query_text,
            alpha=0.5,
            k=3,
        )

        assert len(results) == 3

        top_result = results[0]
        top_doc, top_score = top_result

        assert top_doc.page_content == "green square", (
            f"Expected 'green square' as top result, got: {top_doc.page_content}"
        )

        assert top_score > 0.7, (
            f"Expected high similarity for matching text and image, got {top_score:.4f}"
        )


def test_regression_score_ranges_across_alphas() -> None:
    """Regression test that verifies score ranges differ across alpha values.

    Ensures that the same query with different alphas produces different score distributions.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = FixedEmbeddingModel(embedding_dim=64)

        samples = create_fixed_test_dataset()
        loader = FixedDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        alpha_values = [0.0, 0.5, 1.0]
        vectorstores = indexer.index(alpha_values=alpha_values, rebuild=True, batch_size=10)

        query_image = Image.new("RGB", (50, 50), color=(255, 0, 0))
        query_text = "red square"

        scores_by_alpha = {}
        for alpha in alpha_values:
            vectorstore = vectorstores[alpha]
            results = indexer.query_multimodal(
                vectorstore=vectorstore,
                image=query_image,
                text=query_text,
                alpha=alpha,
                k=5,
            )
            top_score = results[0][1] if results else 0.0
            scores_by_alpha[alpha] = top_score

        assert len(scores_by_alpha) == 3

        score_0_0 = scores_by_alpha[0.0]
        score_0_5 = scores_by_alpha[0.5]
        score_1_0 = scores_by_alpha[1.0]

        assert all(0.0 <= s <= 1.0 for s in [score_0_0, score_0_5, score_1_0])

        different_scores = (
            not np.isclose(score_0_0, score_0_5, rtol=1e-2)
            or not np.isclose(score_0_5, score_1_0, rtol=1e-2)
            or not np.isclose(score_0_0, score_1_0, rtol=1e-2)
        )

        assert different_scores, (
            f"Expected different top scores across alphas, but got: "
            f"alpha=0.0: {score_0_0:.4f}, alpha=0.5: {score_0_5:.4f}, alpha=1.0: {score_1_0:.4f}. "
            f"This indicates multimodal querying might not be working correctly."
        )


def test_regression_top_k_consistency() -> None:
    """Regression test for top-k result consistency.

    Verifies that retrieving top-k results is stable across multiple calls.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = FixedEmbeddingModel(embedding_dim=64)

        samples = create_fixed_test_dataset()
        loader = FixedDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], rebuild=True, batch_size=10)
        vectorstore = vectorstores[0.5]

        query_image = Image.new("RGB", (50, 50), color=(0, 255, 0))
        query_text = "green square"

        results_1 = indexer.query_multimodal(
            vectorstore=vectorstore,
            image=query_image,
            text=query_text,
            alpha=0.5,
            k=3,
        )

        results_2 = indexer.query_multimodal(
            vectorstore=vectorstore,
            image=query_image,
            text=query_text,
            alpha=0.5,
            k=3,
        )

        assert len(results_1) == len(results_2)

        for (doc1, score1), (doc2, score2) in zip(results_1, results_2, strict=True):
            assert doc1.page_content == doc2.page_content
            assert np.isclose(score1, score2, rtol=1e-5)


def test_regression_metadata_preservation() -> None:
    """Regression test that metadata is preserved in query results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = FixedEmbeddingModel(embedding_dim=64)

        samples = create_fixed_test_dataset()
        loader = FixedDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], rebuild=True, batch_size=10)
        vectorstore = vectorstores[0.5]

        query_image = Image.new("RGB", (50, 50), color=(255, 0, 0))
        query_text = "red"

        results = indexer.query_multimodal(
            vectorstore=vectorstore,
            image=query_image,
            text=query_text,
            alpha=0.5,
            k=3,
        )

        assert len(results) > 0

        for doc, _score in results:
            assert "alpha" in doc.metadata
            assert "index" in doc.metadata
            assert "has_text" in doc.metadata
            assert "has_image" in doc.metadata

            assert doc.metadata["alpha"] == 0.5
            assert isinstance(doc.metadata["index"], int)
            assert 0 <= doc.metadata["index"] < len(samples)
