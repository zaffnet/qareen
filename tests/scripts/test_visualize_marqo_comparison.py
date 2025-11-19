"""Tests for visualize_marqo_comparison script."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from typer.testing import CliRunner

from scripts.visualize_marqo_comparison import app, download_image, truncate_text

runner = CliRunner()


def test_truncate_text_short():
    text = "Short text"
    result = truncate_text(text, 20)
    assert result == "Short text"


def test_truncate_text_exact_length():
    text = "Exactly twenty chars"
    result = truncate_text(text, 20)
    assert result == "Exactly twenty chars"


def test_truncate_text_long():
    text = "This is a very long text that needs to be truncated"
    result = truncate_text(text, 20)
    assert result == "This is a very long ..."
    assert len(result) == 23


@patch("scripts.visualize_marqo_comparison.requests.get")
@patch("scripts.visualize_marqo_comparison.Image.open")
def test_download_image_success(mock_image_open, mock_requests_get, tmp_path):
    mock_response = Mock()
    mock_response.content = b"fake_image_data"
    mock_requests_get.return_value = mock_response

    mock_img = Mock()
    mock_img.mode = "RGB"
    mock_image_open.return_value = mock_img

    output_path = tmp_path / "test_image.jpg"
    result = download_image("http://example.com/image.jpg", output_path)

    assert result is True
    mock_requests_get.assert_called_once_with("http://example.com/image.jpg", timeout=30)
    mock_img.save.assert_called_once_with(output_path)


@patch("scripts.visualize_marqo_comparison.requests.get")
@patch("scripts.visualize_marqo_comparison.Image.open")
def test_download_image_convert_to_rgb(mock_image_open, mock_requests_get, tmp_path):
    mock_response = Mock()
    mock_response.content = b"fake_image_data"
    mock_requests_get.return_value = mock_response

    mock_img = Mock()
    mock_img.mode = "RGBA"
    mock_converted_img = Mock()
    mock_img.convert.return_value = mock_converted_img
    mock_image_open.return_value = mock_img

    output_path = tmp_path / "test_image.jpg"
    result = download_image("http://example.com/image.jpg", output_path)

    assert result is True
    mock_img.convert.assert_called_once_with("RGB")
    mock_converted_img.save.assert_called_once_with(output_path)


@patch("scripts.visualize_marqo_comparison.requests.get")
def test_download_image_network_failure(mock_requests_get, tmp_path):
    mock_requests_get.side_effect = Exception("Network error")

    output_path = tmp_path / "test_image.jpg"
    result = download_image("http://example.com/image.jpg", output_path)

    assert result is False


@patch("scripts.visualize_marqo_comparison.load_from_disk")
def test_main_invalid_environment(mock_load_from_disk):
    result = runner.invoke(
        app,
        ["--dataset-path", "data/test", "--environment", "invalid"],
    )

    assert result.exit_code == 1
    mock_load_from_disk.assert_not_called()


@patch("scripts.visualize_marqo_comparison.load_from_disk")
def test_main_empty_dataset(mock_load_from_disk):
    mock_dataset: list[dict] = []
    mock_load_from_disk.return_value = mock_dataset

    result = runner.invoke(
        app,
        ["--dataset-path", "data/test", "--environment", "dev"],
    )

    assert result.exit_code == 1


@patch("scripts.visualize_marqo_comparison.load_from_disk")
def test_main_sample_index_out_of_bounds(mock_load_from_disk):
    mock_dataset = MagicMock()
    type(mock_dataset).__len__ = lambda _: 10
    mock_load_from_disk.return_value = mock_dataset

    result = runner.invoke(
        app,
        ["--dataset-path", "data/test", "--sample-index", "20"],
    )

    assert result.exit_code == 1


@patch("scripts.visualize_marqo_comparison.load_from_disk")
@patch("scripts.visualize_marqo_comparison.LocalDatasetLoader")
@patch("scripts.visualize_marqo_comparison.MarqoFashionSigLIPModel")
@patch("scripts.visualize_marqo_comparison.ChromaIndexer")
@patch("scripts.visualize_marqo_comparison.Settings")
def test_main_success(
    mock_settings,
    mock_indexer_cls,
    mock_model_cls,
    mock_loader_cls,
    mock_load_from_disk,
    tmp_path,
):
    mock_dataset = MagicMock()
    type(mock_dataset).__len__ = lambda _: 100
    mock_dataset.__getitem__ = lambda _, idx: {"text": "test query", "image": None}
    mock_load_from_disk.return_value = mock_dataset

    mock_loader = Mock()
    mock_loader.get_dataset_name.return_value = "test_dataset"
    mock_loader_cls.return_value = mock_loader

    mock_model = Mock()
    mock_model_cls.return_value = mock_model

    mock_vectorstore = Mock()

    mock_indexer = Mock()
    mock_indexer.create_vectorstore.return_value = mock_vectorstore
    mock_indexer.query_multimodal.return_value = []
    mock_indexer_cls.return_value = mock_indexer

    mock_settings.return_value = Mock()

    output_path = str(tmp_path / "output.md")

    result = runner.invoke(
        app,
        [
            "--dataset-path",
            "data/test",
            "--models",
            "Marqo/marqo-fashionSigLIP",
            "--alpha-values",
            "0.0",
            "--alpha-values",
            "0.5",
            "--output",
            output_path,
            "--k",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert Path(output_path).exists()


@patch("scripts.visualize_marqo_comparison.load_from_disk")
@patch("scripts.visualize_marqo_comparison.Settings")
def test_main_exception_during_processing(_mock_settings, mock_load_from_disk):
    mock_load_from_disk.side_effect = Exception("Processing error")

    result = runner.invoke(
        app,
        ["--dataset-path", "data/test"],
    )

    assert result.exit_code == 1
