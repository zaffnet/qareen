"""Tests for HuggingFace dataset loader."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader


def test_init():
    loader = HuggingFaceDatasetLoader(
        dataset_name="test/dataset",
        split="train",
        cache_dir="/tmp/cache",
    )
    assert loader.dataset_name == "test/dataset"
    assert loader.split == "train"
    assert loader.load_kwargs == {"cache_dir": "/tmp/cache"}
    assert loader._dataset is None


@patch("qareen.dataset.hf_dataset.load_dataset")
def test_load(mock_load_dataset):
    mock_dataset = Mock()
    mock_load_dataset.return_value = mock_dataset

    loader = HuggingFaceDatasetLoader(dataset_name="test/dataset", split="train")
    result = loader.load()

    assert result == mock_dataset
    mock_load_dataset.assert_called_once_with("test/dataset", split="train")


@patch("qareen.dataset.hf_dataset.load_dataset")
def test_load_caches(mock_load_dataset):
    mock_dataset = Mock()
    mock_load_dataset.return_value = mock_dataset

    loader = HuggingFaceDatasetLoader(dataset_name="test/dataset")
    loader.load()
    loader.load()

    mock_load_dataset.assert_called_once()


@patch("qareen.dataset.hf_dataset.load_dataset")
@patch("qareen.dataset.hf_dataset.validate_dataset_schema")
def test_validate_schema(mock_validate, mock_load_dataset):
    mock_dataset = Mock()
    mock_load_dataset.return_value = mock_dataset

    loader = HuggingFaceDatasetLoader(dataset_name="test/dataset")
    loader.validate_schema()

    mock_validate.assert_called_once_with(mock_dataset, loader.MISSING_FIELDS_ERROR)


def test_get_dataset_name():
    loader = HuggingFaceDatasetLoader(dataset_name="test/dataset")
    assert loader.get_dataset_name() == "test/dataset"


@patch("qareen.dataset.hf_dataset.load_dataset")
def test_get_dataset_info_single_split(mock_load_dataset):
    mock_dataset = Mock()
    mock_dataset.features = {"text": Mock(), "image": Mock()}
    mock_load_dataset.return_value = mock_dataset
    type(mock_dataset).__len__ = lambda self: 100

    loader = HuggingFaceDatasetLoader(dataset_name="test/dataset", split="train")
    info = loader.get_dataset_info()

    assert info == {
        "dataset_name": "test/dataset",
        "split": "train",
        "num_rows": 100,
        "features": ["text", "image"],
    }


@patch("qareen.dataset.hf_dataset.load_dataset")
def test_get_dataset_info_multiple_splits(mock_load_dataset):
    mock_train = MagicMock()
    mock_train.features = {"text": Mock(), "image": Mock()}
    type(mock_train).__len__ = lambda self: 100

    mock_test = MagicMock()
    mock_test.features = {"text": Mock(), "image": Mock()}
    type(mock_test).__len__ = lambda self: 50

    mock_dataset_dict = {"train": mock_train, "test": mock_test}
    mock_load_dataset.return_value = mock_dataset_dict

    loader = HuggingFaceDatasetLoader(dataset_name="test/dataset")
    info = loader.get_dataset_info()

    assert info == {
        "dataset_name": "test/dataset",
        "splits": ["train", "test"],
        "num_rows": {"train": 100, "test": 50},
        "features": ["text", "image"],
    }


@patch("qareen.dataset.hf_dataset.load_dataset")
def test_get_dataset_info_empty_dict(mock_load_dataset):
    mock_load_dataset.return_value = {}

    loader = HuggingFaceDatasetLoader(dataset_name="test/dataset")
    info = loader.get_dataset_info()

    assert info == {
        "dataset_name": "test/dataset",
        "splits": [],
        "num_rows": {},
        "features": [],
    }
