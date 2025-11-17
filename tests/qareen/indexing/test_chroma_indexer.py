"""ChromaIndexer regression tests."""

from __future__ import annotations

import tempfile
from contextlib import suppress
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from qareen.config.settings import Settings
from qareen.dataset.base import DatasetLoader
from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.models import EmbeddingModel


class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing."""

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
        """Generate deterministic text embedding based on text hash.

        Args:
            text: Input text or None

        Returns:
            L2-normalized embedding vector or None if text is None
        """
        if text is None:
            return None
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        return self.normalize_l2(embedding)

    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        """Generate deterministic image embedding.

        Args:
            image: Input image or None

        Returns:
            L2-normalized embedding vector or None if image is None
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
        """Generate combined multimodal embedding.

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
        return "mock_model"


class MockDatasetLoader(DatasetLoader):
    """Mock dataset loader for testing."""

    def __init__(self, dataset_size: int = 3, track_select: bool = False) -> None:
        """Initialize mock dataset loader.

        Args:
            dataset_size: Size of the dataset to return
            track_select: Whether to track select() calls
        """
        self.dataset_size = dataset_size
        self.track_select = track_select
        self.select_calls: list[range] = []

    def load(self) -> dict | MagicMock:
        """Return mock dataset.

        Returns:
            Mock dataset with text and image fields
        """
        if self.track_select:
            dataset = MagicMock()
            dataset.__len__ = lambda *args: self.dataset_size
            dataset.column_names = ["text", "image"]

            def select(indices: range) -> MagicMock:
                self.select_calls.append(indices)
                indices_list = list(indices)
                selected = MagicMock()
                selected.__len__ = lambda *args: len(indices_list)
                selected.column_names = ["text", "image"]

                def getitem_slice(*args: object) -> dict:
                    slice_obj = args[-1] if args else slice(None)
                    if not isinstance(slice_obj, slice):
                        slice_obj = slice(None)
                    start = slice_obj.start or 0
                    stop = slice_obj.stop or len(indices_list)
                    step = slice_obj.step or 1
                    selected_indices = indices_list[start:stop:step]
                    return {
                        "text": [f"text_{i}" for i in selected_indices],
                        "image": [
                            Image.new("RGB", (224, 224), color="red") for _ in selected_indices
                        ],
                    }

                selected.__getitem__ = getitem_slice
                return selected

            dataset.select = select

            def getitem_slice(*args: object) -> dict:
                slice_obj = args[-1] if args else slice(None)
                if not isinstance(slice_obj, slice):
                    slice_obj = slice(None)
                start = slice_obj.start or 0
                stop = slice_obj.stop or self.dataset_size
                step = slice_obj.step or 1
                return {
                    "text": [f"text_{i}" for i in range(start, stop, step)],
                    "image": [
                        Image.new("RGB", (224, 224), color="red") for _ in range(start, stop, step)
                    ],
                }

            dataset.__getitem__ = getitem_slice
            return dataset

        return {
            "text": ["apple", "banana", "cherry"],
            "image": [Image.new("RGB", (224, 224), color="red") for _ in range(3)],
        }

    def get_dataset_name(self) -> str:
        """Return mock dataset name.

        Returns:
            Dataset identifier
        """
        return "test_dataset"

    def validate_schema(self) -> None:
        """Validate schema."""
        pass

    def get_dataset_info(self) -> dict[str, object]:
        """Return dataset info."""
        return {"size": self.dataset_size}


def test_similarity_search_works_with_embedding_wrapper() -> None:
    """Regression test: similarity_search should work with EmbeddingModelWrapper."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        embedding_model = MockEmbeddingModel(embedding_dim=128)
        dataset_loader = MockDatasetLoader(dataset_size=3, track_select=True)

        indexer = ChromaIndexer(
            dataset_loader=dataset_loader,
            embedding_model=embedding_model,
            settings=settings,
        )

        vectorstores = indexer.index(alpha_values=[0.5], batch_size=10)

        assert len(vectorstores) == 1
        vectorstore = vectorstores[0.5]

        results = vectorstore.similarity_search("apple", k=1)

        assert len(results) == 1
        assert results[0].page_content in ["text_0", "text_1", "text_2"]


def test_create_vectorstore_similarity_search_works() -> None:
    """Test that create_vectorstore returns a usable vectorstore for similarity_search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", chroma_db_dir=Path(tmpdir))
        embedding_model = MockEmbeddingModel(embedding_dim=128)
        dataset_loader = MockDatasetLoader(dataset_size=3, track_select=True)

        indexer = ChromaIndexer(
            dataset_loader=dataset_loader,
            embedding_model=embedding_model,
            settings=settings,
        )

        indexer.index(alpha_values=[0.5], batch_size=10)

        vectorstore = indexer.create_vectorstore(
            dataset_name="test_dataset",
            model_id="mock_model",
            alpha=0.5,
            environment="dev",
        )

        results = vectorstore.similarity_search("banana", k=1)

        assert len(results) == 1
        assert results[0].page_content in ["text_0", "text_1", "text_2"]


def test_sample_size_honored_in_non_dev_environment() -> None:
    """Test that sample_size argument is honored in non-dev environments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="prod", chroma_db_dir=Path(tmpdir))
        embedding_model = MockEmbeddingModel(embedding_dim=128)
        dataset_loader = MockDatasetLoader(dataset_size=100, track_select=True)

        indexer = ChromaIndexer(
            dataset_loader=dataset_loader,
            embedding_model=embedding_model,
            settings=settings,
        )

        with suppress(Exception):
            indexer.index(alpha_values=[0.5], batch_size=10, sample_size=5)

        assert len(dataset_loader.select_calls) == 1
        select_call = dataset_loader.select_calls[0]
        assert len(select_call) == 5


def test_sample_size_honored_in_staging_environment() -> None:
    """Test that sample_size argument is honored in staging environment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="staging", chroma_db_dir=Path(tmpdir))
        embedding_model = MockEmbeddingModel(embedding_dim=128)
        dataset_loader = MockDatasetLoader(dataset_size=100, track_select=True)

        indexer = ChromaIndexer(
            dataset_loader=dataset_loader,
            embedding_model=embedding_model,
            settings=settings,
        )

        with suppress(Exception):
            indexer.index(alpha_values=[0.5], batch_size=10, sample_size=7)

        assert len(dataset_loader.select_calls) == 1
        select_call = dataset_loader.select_calls[0]
        assert len(select_call) == 7


def test_dev_sample_size_fallback_in_dev_environment() -> None:
    """Test that dev_sample_size is used when sample_size is None in dev."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", dev_sample_size=42, chroma_db_dir=Path(tmpdir))
        embedding_model = MockEmbeddingModel(embedding_dim=128)
        dataset_loader = MockDatasetLoader(dataset_size=100, track_select=True)

        indexer = ChromaIndexer(
            dataset_loader=dataset_loader,
            embedding_model=embedding_model,
            settings=settings,
        )

        with suppress(Exception):
            indexer.index(alpha_values=[0.5], batch_size=10, sample_size=None)

        assert len(dataset_loader.select_calls) == 1
        select_call = dataset_loader.select_calls[0]
        assert len(select_call) == 42


def test_explicit_sample_size_overrides_dev_sample_size() -> None:
    """Test that explicit sample_size overrides dev_sample_size in dev."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="dev", dev_sample_size=100, chroma_db_dir=Path(tmpdir))
        embedding_model = MockEmbeddingModel(embedding_dim=128)
        dataset_loader = MockDatasetLoader(dataset_size=200, track_select=True)

        indexer = ChromaIndexer(
            dataset_loader=dataset_loader,
            embedding_model=embedding_model,
            settings=settings,
        )

        with suppress(Exception):
            indexer.index(alpha_values=[0.5], batch_size=10, sample_size=15)

        assert len(dataset_loader.select_calls) == 1
        select_call = dataset_loader.select_calls[0]
        assert len(select_call) == 15


def test_no_limit_in_non_dev_when_sample_size_none() -> None:
    """Test that no limit is applied in non-dev when sample_size is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(environment="prod", chroma_db_dir=Path(tmpdir))
        embedding_model = MockEmbeddingModel(embedding_dim=128)
        dataset_loader = MockDatasetLoader(dataset_size=100, track_select=True)

        indexer = ChromaIndexer(
            dataset_loader=dataset_loader,
            embedding_model=embedding_model,
            settings=settings,
        )

        with suppress(Exception):
            indexer.index(alpha_values=[0.5], batch_size=10, sample_size=None)

        assert len(dataset_loader.select_calls) == 0
