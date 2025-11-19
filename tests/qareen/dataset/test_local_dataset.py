"""Tests for local dataset loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from qareen.dataset.local_dataset import LocalDatasetLoader


def test_init_with_string():
    loader = LocalDatasetLoader(dataset_path="data/test")
    assert loader.dataset_path == Path("data/test")
    assert loader._dataset is None


def test_init_with_path():
    loader = LocalDatasetLoader(dataset_path=Path("data/test"))
    assert loader.dataset_path == Path("data/test")
    assert loader._dataset is None


@patch("qareen.dataset.local_dataset.load_from_disk")
def test_load(mock_load_from_disk):
    mock_dataset = Mock()
    mock_load_from_disk.return_value = mock_dataset

    loader = LocalDatasetLoader(dataset_path="data/test")
    result = loader.load()

    assert result == mock_dataset
    mock_load_from_disk.assert_called_once_with("data/test")


@patch("qareen.dataset.local_dataset.load_from_disk")
def test_load_caches(mock_load_from_disk):
    mock_dataset = Mock()
    mock_load_from_disk.return_value = mock_dataset

    loader = LocalDatasetLoader(dataset_path="data/test")
    loader.load()
    loader.load()

    mock_load_from_disk.assert_called_once()


@patch("qareen.dataset.local_dataset.load_from_disk")
@patch("qareen.dataset.local_dataset.validate_dataset_schema")
def test_validate_schema(mock_validate, mock_load_from_disk):
    mock_dataset = Mock()
    mock_load_from_disk.return_value = mock_dataset

    loader = LocalDatasetLoader(dataset_path="data/test")
    loader.validate_schema()

    mock_validate.assert_called_once_with(mock_dataset, loader.MISSING_FIELDS_ERROR)


def test_get_dataset_name():
    loader = LocalDatasetLoader(dataset_path="data/my_dataset")
    assert loader.get_dataset_name() == "my_dataset"


@patch("qareen.dataset.local_dataset.load_from_disk")
def test_get_dataset_info_single_split(mock_load_from_disk):
    mock_dataset = Mock()
    mock_dataset.features = {"text": Mock(), "image": Mock()}
    mock_load_from_disk.return_value = mock_dataset
    type(mock_dataset).__len__ = lambda self: 100

    loader = LocalDatasetLoader(dataset_path="data/test_dataset")
    info = loader.get_dataset_info()

    assert info == {
        "dataset_name": "test_dataset",
        "num_rows": 100,
        "features": ["text", "image"],
    }


@patch("qareen.dataset.local_dataset.load_from_disk")
def test_get_dataset_info_multiple_splits(mock_load_from_disk):
    mock_train = MagicMock()
    mock_train.features = {"text": Mock(), "image": Mock()}
    type(mock_train).__len__ = lambda self: 100

    mock_test = MagicMock()
    mock_test.features = {"text": Mock(), "image": Mock()}
    type(mock_test).__len__ = lambda self: 50

    mock_dataset_dict = {"train": mock_train, "test": mock_test}
    mock_load_from_disk.return_value = mock_dataset_dict

    loader = LocalDatasetLoader(dataset_path="data/test_dataset")
    info = loader.get_dataset_info()

    assert info == {
        "dataset_name": "test_dataset",
        "splits": ["train", "test"],
        "num_rows": {"train": 100, "test": 50},
        "features": ["text", "image"],
    }


@patch("qareen.dataset.local_dataset.load_from_disk")
def test_get_dataset_info_empty_dict(mock_load_from_disk):
    mock_load_from_disk.return_value = {}

    loader = LocalDatasetLoader(dataset_path="data/test")
    info = loader.get_dataset_info()

    assert info == {
        "dataset_name": "test",
        "splits": [],
        "num_rows": {},
        "features": [],
    }
