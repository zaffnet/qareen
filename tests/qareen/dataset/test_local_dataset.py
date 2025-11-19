"""Tests for local dataset loader."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from qareen.dataset.local_dataset import LocalDatasetLoader


def test_init_with_string(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    loader = LocalDatasetLoader(dataset_path=str(test_dir))
    assert loader.dataset_path == test_dir
    assert loader._dataset is None


def test_init_with_path(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    loader = LocalDatasetLoader(dataset_path=test_dir)
    assert loader.dataset_path == test_dir
    assert loader._dataset is None


def test_init_path_not_exists():
    with pytest.raises(ValueError, match="Dataset path does not exist"):
        LocalDatasetLoader(dataset_path="nonexistent/path")


def test_init_path_not_directory(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("not a directory")
    with pytest.raises(ValueError, match="Dataset path is not a directory"):
        LocalDatasetLoader(dataset_path=test_file)


@patch("qareen.dataset.local_dataset.load_from_disk")
def test_load(mock_load_from_disk, tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    mock_dataset = Mock()
    mock_load_from_disk.return_value = mock_dataset

    loader = LocalDatasetLoader(dataset_path=test_dir)
    result = loader.load()

    assert result == mock_dataset
    mock_load_from_disk.assert_called_once_with(str(test_dir))


@patch("qareen.dataset.local_dataset.load_from_disk")
def test_load_caches(mock_load_from_disk, tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    mock_dataset = Mock()
    mock_load_from_disk.return_value = mock_dataset

    loader = LocalDatasetLoader(dataset_path=test_dir)
    loader.load()
    loader.load()

    mock_load_from_disk.assert_called_once()


@patch("qareen.dataset.local_dataset.load_from_disk")
@patch("qareen.dataset.local_dataset.validate_dataset_schema")
def test_validate_schema(mock_validate, mock_load_from_disk, tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    mock_dataset = Mock()
    mock_load_from_disk.return_value = mock_dataset

    loader = LocalDatasetLoader(dataset_path=test_dir)
    loader.validate_schema()

    mock_validate.assert_called_once_with(mock_dataset, loader.MISSING_FIELDS_ERROR)


def test_get_dataset_name(tmp_path):
    test_dir = tmp_path / "my_dataset"
    test_dir.mkdir()
    loader = LocalDatasetLoader(dataset_path=test_dir)
    assert loader.get_dataset_name() == "my_dataset"


@patch("qareen.dataset.local_dataset.load_from_disk")
def test_get_dataset_info_single_split(mock_load_from_disk, tmp_path):
    test_dir = tmp_path / "test_dataset"
    test_dir.mkdir()
    mock_dataset = Mock()
    mock_dataset.features = {"text": Mock(), "image": Mock()}
    mock_load_from_disk.return_value = mock_dataset
    type(mock_dataset).__len__ = lambda _: 100

    loader = LocalDatasetLoader(dataset_path=test_dir)
    info = loader.get_dataset_info()

    assert info == {
        "dataset_name": "test_dataset",
        "num_rows": 100,
        "features": ["text", "image"],
    }


@patch("qareen.dataset.local_dataset.load_from_disk")
def test_get_dataset_info_multiple_splits(mock_load_from_disk, tmp_path):
    test_dir = tmp_path / "test_dataset"
    test_dir.mkdir()
    mock_train = MagicMock()
    mock_train.features = {"text": Mock(), "image": Mock()}
    type(mock_train).__len__ = lambda _: 100

    mock_test = MagicMock()
    mock_test.features = {"text": Mock(), "image": Mock()}
    type(mock_test).__len__ = lambda _: 50

    mock_dataset_dict = {"train": mock_train, "test": mock_test}
    mock_load_from_disk.return_value = mock_dataset_dict

    loader = LocalDatasetLoader(dataset_path=test_dir)
    info = loader.get_dataset_info()

    assert info == {
        "dataset_name": "test_dataset",
        "splits": ["train", "test"],
        "num_rows": {"train": 100, "test": 50},
        "features": ["text", "image"],
    }


@patch("qareen.dataset.local_dataset.load_from_disk")
def test_get_dataset_info_empty_dict(mock_load_from_disk, tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    mock_load_from_disk.return_value = {}

    loader = LocalDatasetLoader(dataset_path=test_dir)
    info = loader.get_dataset_info()

    assert info == {
        "dataset_name": "test",
        "splits": [],
        "num_rows": {},
        "features": [],
    }
