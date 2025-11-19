"""Tests for prepare_marqo_dataset script."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from scripts.prepare_marqo_dataset import app

runner = CliRunner()


@patch("scripts.prepare_marqo_dataset.load_dataset")
def test_main_success(mock_load_dataset, tmp_path):
    mock_dataset = MagicMock()
    mock_dataset.column_names = ["query", "image"]
    type(mock_dataset).__len__ = lambda self: 5000

    mock_dataset.rename_column.return_value = mock_dataset
    mock_dataset.shuffle.return_value = mock_dataset
    mock_dataset.select.return_value = mock_dataset

    mock_load_dataset.return_value = mock_dataset

    output_dir = str(tmp_path / "test_output")

    result = runner.invoke(
        app,
        ["--output-dir", output_dir, "--sample-size", "100", "--seed", "42"],
    )

    assert result.exit_code == 0
    mock_load_dataset.assert_called_once_with("Marqo/marqo-gs-woman-fashion", split="zero_shot")
    mock_dataset.rename_column.assert_called_once_with("query", "text")
    mock_dataset.shuffle.assert_called_once_with(seed=42)
    mock_dataset.save_to_disk.assert_called_once()


@patch("scripts.prepare_marqo_dataset.load_dataset")
def test_main_missing_query_column(mock_load_dataset):
    mock_dataset = MagicMock()
    mock_dataset.column_names = ["image"]
    type(mock_dataset).__len__ = lambda self: 100
    mock_load_dataset.return_value = mock_dataset

    result = runner.invoke(app, ["--output-dir", "test_output"])

    assert result.exit_code == 0


@patch("scripts.prepare_marqo_dataset.load_dataset")
def test_main_missing_image_column(mock_load_dataset):
    mock_dataset = MagicMock()
    mock_dataset.column_names = ["query"]
    type(mock_dataset).__len__ = lambda self: 100
    mock_load_dataset.return_value = mock_dataset

    result = runner.invoke(app, ["--output-dir", "test_output"])

    assert result.exit_code == 0


@patch("scripts.prepare_marqo_dataset.load_dataset")
def test_main_dataset_smaller_than_sample_size(mock_load_dataset, tmp_path):
    mock_dataset = MagicMock()
    mock_dataset.column_names = ["query", "image"]
    type(mock_dataset).__len__ = lambda self: 50

    mock_dataset.rename_column.return_value = mock_dataset
    mock_dataset.shuffle.return_value = mock_dataset
    mock_dataset.select.return_value = mock_dataset

    mock_load_dataset.return_value = mock_dataset

    output_dir = str(tmp_path / "test_output")

    result = runner.invoke(
        app,
        ["--output-dir", output_dir, "--sample-size", "100", "--seed", "42"],
    )

    assert result.exit_code == 0
    mock_dataset.select.assert_called_once_with(range(50))


@patch("scripts.prepare_marqo_dataset.load_dataset")
def test_main_load_dataset_exception(mock_load_dataset):
    mock_load_dataset.side_effect = Exception("Network error")

    result = runner.invoke(app, ["--output-dir", "test_output"])

    assert result.exit_code == 0


def test_main_default_arguments():
    with patch("scripts.prepare_marqo_dataset.load_dataset") as mock_load_dataset:
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["query", "image"]
        type(mock_dataset).__len__ = lambda self: 5000

        mock_dataset.rename_column.return_value = mock_dataset
        mock_dataset.shuffle.return_value = mock_dataset
        mock_dataset.select.return_value = mock_dataset

        mock_load_dataset.return_value = mock_dataset

        result = runner.invoke(app, [])

        assert result.exit_code == 0
        mock_dataset.shuffle.assert_called_once_with(seed=42)
        mock_dataset.select.assert_called_once_with(range(3000))
