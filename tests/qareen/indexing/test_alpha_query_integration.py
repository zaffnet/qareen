"""Integration tests for alpha-weighted multimodal retrieval.

Tests that verify different alpha values produce meaningfully different
retrieval results when querying with multimodal inputs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from PIL import Image

from qareen.dataset.base import DatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.embedding_model import EmbeddingModel
from qareen.models import Settings
from qareen.retrieving.chroma_retriever import ChromaRetriever


class DeterministicEmbeddingModel(EmbeddingModel):
    """Deterministic embedding model for reproducible integration tests."""

    def __init__(self, embedding_dim: int = 256) -> None:
        """Initialize model.

        Args:
            embedding_dim: Dimension of embeddings

        """
        self._embedding_dim = embedding_dim
        self._model_loaded = False

    def load_model(self) -> None:
        """Load model (mock)."""
        self._model_loaded = True

    def embed_text(self, text: str | None) -> np.ndarray | None:
        """Generate deterministic text embedding based on text content.

        Args:
            text: Input text

        Returns:
            Embedding or None if text is None

        """
        if text is None:
            return None

        seed = abs(hash(text)) % (2**31)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self._embedding_dim).astype(np.float32)
        return self.normalize_l2(embedding)

    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        """Generate deterministic image embedding based on pixel values.

        Args:
            image: Input image

        Returns:
            Embedding or None if image is None

        """
        if image is None:
            return None

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if not isinstance(image, Image.Image):
            raise TypeError("Image must be PIL Image or path string")

        image_array = np.array(image)
        seed = int(image_array.sum() + image_array.mean() * 1000) % (2**31)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self._embedding_dim).astype(np.float32)
        return self.normalize_l2(embedding)

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
        return "deterministic_test_model"

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension.

        Returns:
            Embedding dimension

        """
        return self._embedding_dim


class TestDatasetLoader(DatasetLoader):
    """Dataset loader for integration tests."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        """Initialize with sample data.

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
        """Validate dataset schema."""

    def get_dataset_name(self) -> str:
        """Return dataset name.

        Returns:
            Dataset name

        """
        return "integration_test_dataset"

    def get_dataset_info(self) -> dict[str, object]:
        """Return dataset info.

        Returns:
            Dataset info

        """
        return {"size": len(self.samples)}


def test_alpha_spectrum_produces_different_results() -> None:
    """Test that querying across alpha spectrum produces meaningfully different results.

    This integration test creates a dataset where text and image embeddings
    would retrieve different items, then verifies that alpha values influence
    which items are retrieved.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = DeterministicEmbeddingModel(embedding_dim=256)

        img_red = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img_green = Image.new("RGB", (100, 100), color=(0, 255, 0))
        img_blue = Image.new("RGB", (100, 100), color=(0, 0, 255))

        samples = [
            {"text": "category_A item_1", "image": img_red},
            {"text": "category_A item_2", "image": img_green},
            {"text": "category_B item_3", "image": img_blue},
            {"text": "category_A item_4", "image": img_blue},
            {"text": "category_B item_5", "image": img_red},
        ]

        loader = TestDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        alpha_values = [0.0, 0.5, 1.0]
        vectorstores = indexer.index(alpha_values=alpha_values, rebuild=True, batch_size=10)

        retriever = ChromaRetriever(model, settings)

        query_image = Image.new("RGB", (100, 100), color=(255, 0, 0))
        query_text = "category_A"

        results_by_alpha = {}
        for alpha in alpha_values:
            vectorstore = vectorstores[alpha]
            results = retriever.query_multimodal(
                vectorstore=vectorstore,
                image=query_image,
                text=query_text,
                alpha=alpha,
                k=3,
            )
            results_by_alpha[alpha] = results

        results_text_only = results_by_alpha[0.0]
        results_balanced = results_by_alpha[0.5]
        results_image_only = results_by_alpha[1.0]

        assert len(results_text_only) > 0
        assert len(results_balanced) > 0
        assert len(results_image_only) > 0

        top_indices_text = [r[0].metadata.get("index") for r in results_text_only]
        top_indices_image = [r[0].metadata.get("index") for r in results_image_only]

        scores_text = [r[1] for r in results_text_only]
        scores_image = [r[1] for r in results_image_only]

        different_results = top_indices_text != top_indices_image or not all(
            np.isclose(s1, s2, rtol=1e-3) for s1, s2 in zip(scores_text, scores_image, strict=False)
        )
        assert different_results, (
            "Alpha 0.0 (text-only) and alpha 1.0 (image-only) should produce "
            "different results or different scores"
        )


def test_alpha_affects_score_distribution() -> None:
    """Test that different alpha values produce different score distributions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = DeterministicEmbeddingModel(embedding_dim=128)

        img1 = Image.new("RGB", (50, 50), color=(200, 50, 50))
        img2 = Image.new("RGB", (50, 50), color=(50, 200, 50))
        img3 = Image.new("RGB", (50, 50), color=(50, 50, 200))

        samples = [
            {"text": "red object", "image": img1},
            {"text": "green object", "image": img2},
            {"text": "blue object", "image": img3},
            {"text": "red item", "image": img2},
            {"text": "blue thing", "image": img1},
        ]

        loader = TestDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        vectorstores = indexer.index(alpha_values=alpha_values, rebuild=True, batch_size=10)

        retriever = ChromaRetriever(model, settings)

        query_image = Image.new("RGB", (50, 50), color=(200, 50, 50))
        query_text = "red object"

        scores_by_alpha = {}
        for alpha in alpha_values:
            vectorstore = vectorstores[alpha]
            results = retriever.query_multimodal(
                vectorstore=vectorstore,
                image=query_image,
                text=query_text,
                alpha=alpha,
                k=5,
            )
            scores = [score for _, score in results]
            scores_by_alpha[alpha] = scores

        assert len(scores_by_alpha) == len(alpha_values)

        for alpha, scores in scores_by_alpha.items():
            assert len(scores) > 0, f"Should have results for alpha={alpha}"
            assert all(0.0 <= s <= 1.0 for s in scores), f"Scores out of range for alpha={alpha}"

        score_means = {alpha: np.mean(scores) for alpha, scores in scores_by_alpha.items()}

        unique_means = len({round(m, 4) for m in score_means.values()})
        assert unique_means >= 3, (
            f"Expected diverse score distributions across alphas, "
            f"but got similar means: {score_means}"
        )


def test_extreme_alphas_behave_differently() -> None:
    """Test that alpha=0.0 and alpha=1.0 produce clearly different results.

    Alpha 0.0 should rely purely on text, alpha 1.0 purely on image.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = DeterministicEmbeddingModel(embedding_dim=256)

        img_similar_to_query = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img_different_from_query = Image.new("RGB", (100, 100), color=(0, 0, 255))

        samples = [
            {"text": "matching text content", "image": img_different_from_query},
            {"text": "unrelated text content", "image": img_similar_to_query},
            {"text": "other text", "image": img_different_from_query},
        ]

        loader = TestDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.0, 1.0], rebuild=True, batch_size=10)

        retriever = ChromaRetriever(model, settings)

        query_image = Image.new("RGB", (100, 100), color=(255, 0, 0))
        query_text = "matching text content"

        results_text_only = retriever.query_multimodal(
            vectorstore=vectorstores[0.0],
            image=query_image,
            text=query_text,
            alpha=0.0,
            k=3,
        )

        results_image_only = retriever.query_multimodal(
            vectorstore=vectorstores[1.0],
            image=query_image,
            text=query_text,
            alpha=1.0,
            k=3,
        )

        top_text_only = results_text_only[0] if results_text_only else None
        top_image_only = results_image_only[0] if results_image_only else None

        assert top_text_only is not None
        assert top_image_only is not None

        text_only_doc = top_text_only[0].page_content
        image_only_doc = top_image_only[0].page_content

        assert text_only_doc != image_only_doc or top_text_only[1] != top_image_only[1], (
            "Text-only and image-only queries should retrieve different documents or "
            "assign different scores when text and image point to different items"
        )


def test_mid_range_alpha_balances_modalities() -> None:
    """Test that mid-range alpha values balance text and image influence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = DeterministicEmbeddingModel(embedding_dim=256)

        img1 = Image.new("RGB", (80, 80), color=(100, 0, 0))
        img2 = Image.new("RGB", (80, 80), color=(0, 100, 0))
        img3 = Image.new("RGB", (80, 80), color=(0, 0, 100))

        samples = [
            {"text": "text_A", "image": img1},
            {"text": "text_B", "image": img2},
            {"text": "text_C", "image": img3},
        ]

        loader = TestDatasetLoader(samples)

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.0, 0.5, 1.0], rebuild=True, batch_size=10)

        retriever = ChromaRetriever(model, settings)

        query_image = Image.new("RGB", (80, 80), color=(100, 0, 0))
        query_text = "text_A"

        results_0 = retriever.query_multimodal(
            vectorstore=vectorstores[0.0],
            image=query_image,
            text=query_text,
            alpha=0.0,
            k=1,
        )

        results_05 = retriever.query_multimodal(
            vectorstore=vectorstores[0.5],
            image=query_image,
            text=query_text,
            alpha=0.5,
            k=1,
        )

        results_1 = retriever.query_multimodal(
            vectorstore=vectorstores[1.0],
            image=query_image,
            text=query_text,
            alpha=1.0,
            k=1,
        )

        score_0 = results_0[0][1] if results_0 else -1.0
        score_05 = results_05[0][1] if results_05 else -1.0
        score_1 = results_1[0][1] if results_1 else -1.0

        assert score_0 >= 0.0, f"Expected valid score for alpha=0.0, got {score_0}"
        assert score_05 >= 0.0, f"Expected valid score for alpha=0.5, got {score_05}"
        assert score_1 >= 0.0, f"Expected valid score for alpha=1.0, got {score_1}"

        all_similar = np.isclose(score_0, score_05, rtol=1e-3) and np.isclose(
            score_05, score_1, rtol=1e-3
        )
        assert not all_similar, (
            f"Expected different scores for different alphas, but got similar values: "
            f"alpha=0.0: {score_0:.4f}, alpha=0.5: {score_05:.4f}, alpha=1.0: {score_1:.4f}"
        )
