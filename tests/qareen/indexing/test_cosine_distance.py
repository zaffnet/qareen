"""Tests for cosine distance implementation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from PIL import Image

from conftest import create_test_settings
from qareen.dataset.base import DatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.embedding_model import EmbeddingModel
from qareen.retrieving.chroma_retriever import ChromaRetriever


class SimpleDatasetLoader(DatasetLoader):
    """Simple dataset loader for testing."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples
        self._dataset: Dataset | None = None

    def load(self) -> Dataset:
        if self._dataset is None:
            self._dataset = Dataset.from_list(self.samples)
        return self._dataset

    def validate_schema(self) -> None:
        pass

    def get_dataset_name(self) -> str:
        return "test_dataset"

    def get_dataset_info(self) -> dict:
        return {"num_samples": len(self.samples)}


class TestEmbeddingModel(EmbeddingModel):
    """Test embedding model with predictable outputs."""

    def __init__(self, embedding_dim: int = 128) -> None:
        self._embedding_dim = embedding_dim
        self.model_loaded = False

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def load_model(self) -> None:
        self.model_loaded = True

    def embed_text(self, text: str | None) -> np.ndarray | None:
        if text is None:
            return None
        rng = np.random.default_rng(hash(text) % 2**32)
        embedding = rng.standard_normal(self.embedding_dim).astype(np.float32)
        return self.normalize_l2(embedding)

    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        if image is None:
            return None
        image_hash = hash(image.tobytes()) if isinstance(image, Image.Image) else hash(str(image))
        rng = np.random.default_rng(image_hash % 2**32)
        embedding = rng.standard_normal(self.embedding_dim).astype(np.float32)
        return self.normalize_l2(embedding)

    def embed_multimodal(
        self,
        image: Image.Image | str | Path | None,
        text: str | None,
        alpha: float,
    ) -> np.ndarray:
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
        return "test_model"


def test_collection_uses_cosine_distance() -> None:
    """Verify ChromaDB collections are created with cosine distance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = create_test_settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = TestEmbeddingModel(embedding_dim=128)
        samples = [{"text": "sample", "image": Image.new("RGB", (10, 10))}]
        loader = SimpleDatasetLoader(samples)
        loader.load()

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        indexer.index(alpha_values=[0.5], rebuild=True, batch_size=10)

        retriever = ChromaRetriever(model, settings)
        vectorstore = retriever.get_vectorstore(
            dataset_name="test_dataset",
            model_id="test_model",
            alpha=0.5,
            environment="dev",
        )

        metadata = vectorstore._collection.metadata
        assert metadata.get("hnsw:space") == "cosine", "Collection must use cosine distance"


def test_score_range_with_cosine() -> None:
    """Verify scores are in [0, 1] range with cosine distance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = create_test_settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = TestEmbeddingModel(embedding_dim=128)

        samples = [
            {"text": "red", "image": Image.new("RGB", (10, 10), color=(255, 0, 0))},
            {"text": "green", "image": Image.new("RGB", (10, 10), color=(0, 255, 0))},
            {"text": "blue", "image": Image.new("RGB", (10, 10), color=(0, 0, 255))},
        ]
        loader = SimpleDatasetLoader(samples)
        loader.load()

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.0, 0.5, 1.0], rebuild=True, batch_size=10)

        retriever = ChromaRetriever(model, settings)

        query_image = Image.new("RGB", (10, 10), color=(255, 0, 0))
        query_text = "red"

        for alpha in [0.0, 0.5, 1.0]:
            results = retriever.query_multimodal(
                vectorstore=vectorstores[alpha],
                image=query_image,
                text=query_text,
                alpha=alpha,
                k=3,
            )

            assert len(results) > 0, f"Should return results for alpha={alpha}"

            for _doc, score in results:
                assert 0.0 <= score <= 1.0, f"Score {score} out of range for alpha={alpha}"


def test_alpha_one_returns_nonzero_scores() -> None:
    """Verify alpha=1.0 returns non-zero scores with cosine distance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = create_test_settings(environment="dev", chroma_db_dir=Path(tmpdir))
        model = TestEmbeddingModel(embedding_dim=256)

        samples = [
            {"text": "item1", "image": Image.new("RGB", (20, 20), color=(100, 0, 0))},
            {"text": "item2", "image": Image.new("RGB", (20, 20), color=(0, 100, 0))},
            {"text": "item3", "image": Image.new("RGB", (20, 20), color=(0, 0, 100))},
        ]
        loader = SimpleDatasetLoader(samples)
        loader.load()

        indexer = ChromaIndexer(
            dataset_loader=loader,
            embedding_model=model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[1.0], rebuild=True, batch_size=10)

        retriever = ChromaRetriever(model, settings)

        query_image = Image.new("RGB", (20, 20), color=(100, 0, 0))
        results = retriever.query_multimodal(
            vectorstore=vectorstores[1.0],
            image=query_image,
            text="query text ignored at alpha=1.0",
            alpha=1.0,
            k=3,
        )

        assert len(results) > 0, "Should return results for alpha=1.0"

        nonzero_count = sum(1 for _, score in results if score > 0.0)
        assert nonzero_count > 0, "At least one result should have non-zero score at alpha=1.0"


def test_l2_normalized_embeddings_required() -> None:
    """Verify embeddings are L2-normalized (required for cosine distance)."""
    model = TestEmbeddingModel(embedding_dim=128)
    model.load_model()

    text_emb = model.embed_text("test")
    assert text_emb is not None
    text_norm = float(np.linalg.norm(text_emb))
    assert np.isclose(text_norm, 1.0, atol=1e-6), f"Text embedding not normalized: {text_norm}"

    image = Image.new("RGB", (10, 10))
    image_emb = model.embed_image(image)
    assert image_emb is not None
    image_norm = float(np.linalg.norm(image_emb))
    assert np.isclose(image_norm, 1.0, atol=1e-6), f"Image embedding not normalized: {image_norm}"

    multimodal_emb = model.embed_multimodal(image=image, text="test", alpha=0.5)
    multimodal_norm = float(np.linalg.norm(multimodal_emb))
    assert np.isclose(multimodal_norm, 1.0, atol=1e-6), (
        f"Multimodal embedding not normalized: {multimodal_norm}"
    )
