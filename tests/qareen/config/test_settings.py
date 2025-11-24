"""Configuration settings behavior."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

from conftest import create_test_settings
from qareen.models import Settings


def test_settings_defaults_match_plan() -> None:
    """
    Verify the Settings model's defaults and types match the expected configuration plan.
    
    Checks that:
    - Settings is a subclass of BaseSettings.
    - `embedding_models` contains at least one string.
    - `data_dir` equals Path("data").
    - `chroma_db_dir` equals Path("chroma_db").
    - `dev_sample_size` equals 300.
    - `environment` is one of "dev", "staging", or "prod".
    """
    assert issubclass(Settings, BaseSettings)

    settings = create_test_settings()
    assert settings.embedding_models, "At least one embedding model is required"
    assert all(isinstance(model, str) for model in settings.embedding_models)

    assert settings.data_dir == Path("data")
    assert settings.chroma_db_dir == Path("chroma_db")
    assert settings.dev_sample_size == 300
    assert settings.environment in {"dev", "staging", "prod"}


def test_ensure_directories_creates_directories(tmp_path: Path) -> None:
    settings = create_test_settings(
        data_dir=tmp_path / "data",
        chroma_db_dir=tmp_path / "chroma_db",
    )

    assert settings.data_dir.exists()
    assert settings.chroma_db_dir.exists()
    assert settings._dirs_ensured


def test_ensure_directories_idempotent(tmp_path: Path) -> None:
    """
    Verify that calling `ensure_directories()` multiple times is idempotent: it sets the internal `_dirs_ensured` flag and ensures both `data_dir` and `chroma_db_dir` exist.
    
    Parameters:
        tmp_path (Path): Temporary filesystem path provided by pytest.
    """
    settings = create_test_settings(
        data_dir=tmp_path / "data",
        chroma_db_dir=tmp_path / "chroma_db",
    )

    settings.ensure_directories()
    settings.ensure_directories()
    settings.ensure_directories()

    assert settings._dirs_ensured
    assert settings.data_dir.exists()
    assert settings.chroma_db_dir.exists()


def test_ensure_directories_with_existing_directories(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    chroma_db_dir = tmp_path / "chroma_db"
    data_dir.mkdir(parents=True)
    chroma_db_dir.mkdir(parents=True)

    settings = create_test_settings(
        data_dir=data_dir,
        chroma_db_dir=chroma_db_dir,
    )

    assert settings.data_dir.exists()
    assert settings.chroma_db_dir.exists()
    assert settings._dirs_ensured

    settings.ensure_directories()

    assert settings._dirs_ensured


def test_ensure_directories_with_partial_existing_directories(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    chroma_db_dir = tmp_path / "chroma_db"

    settings = create_test_settings(
        data_dir=data_dir,
        chroma_db_dir=chroma_db_dir,
    )

    assert settings.data_dir.exists()
    assert settings.chroma_db_dir.exists()
    assert settings._dirs_ensured