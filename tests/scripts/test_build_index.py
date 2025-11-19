"""CLI tests for scripts.build_index."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

import scripts.build_index


def test_build_index_imports() -> None:
    """Test that build_index module imports successfully."""
    assert hasattr(scripts.build_index, "app")
    assert hasattr(scripts.build_index, "main")


def test_build_index_app_is_typer() -> None:
    """Test that app is a Typer instance."""
    assert isinstance(scripts.build_index.app, typer.Typer)


def test_build_index_cli_invalid_environment() -> None:
    """Test CLI with invalid environment completes without crashing."""
    runner = CliRunner()
    result = runner.invoke(
        scripts.build_index.app,
        ["--environment", "invalid_env"],
    )
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_build_index_cli_with_mocked_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI with mocked dataset loader and indexer."""
    runner = CliRunner()

    mock_settings = MagicMock()
    mock_settings.environment = "dev"
    mock_settings.embedding_models = ["test/model"]
    mock_settings.alpha_values = [0.5]
    mock_settings.dev_sample_size = 100
    mock_settings.ensure_directories = MagicMock()

    mock_loader = MagicMock()
    mock_loader.get_dataset_name.return_value = "test_dataset"

    mock_model = MagicMock()
    mock_model.get_model_id.return_value = "test/model"

    mock_indexer = MagicMock()
    mock_indexer.index.return_value = {0.5: MagicMock()}
    mock_indexer.get_collection_name.return_value = "dev_test_dataset_test_model_alpha0_50"

    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_path.is_dir.return_value = True

    with (
        patch("scripts.build_index.Settings", return_value=mock_settings),
        patch("scripts.build_index.Path", return_value=mock_path),
        patch("scripts.build_index.LocalDatasetLoader", return_value=mock_loader),
        patch("scripts.build_index.SIGLIPEmbeddingModel", return_value=mock_model),
        patch("scripts.build_index.ChromaIndexer", return_value=mock_indexer),
    ):
        result = runner.invoke(
            scripts.build_index.app,
            [
                "--dataset-name",
                "data/test_dataset",
                "--environment",
                "dev",
            ],
        )

    assert result.exit_code == 0
    mock_loader.load.assert_called_once()
    mock_loader.validate_schema.assert_called_once()
    mock_indexer.index.assert_called_once()

    call_args = mock_indexer.index.call_args
    assert call_args is not None
    assert call_args.kwargs.get("rebuild") is False
