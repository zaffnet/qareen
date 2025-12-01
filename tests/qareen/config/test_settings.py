"""Configuration settings behavior."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

from conftest import create_test_settings
from qareen.models import Settings


def test_settings_defaults_match_plan(tmp_path: Path) -> None:
    assert issubclass(Settings, BaseSettings)

    # Use Settings directly to test actual defaults
    settings = Settings(
        data_dir=tmp_path / "data",
        chroma_db_dir=tmp_path / "chroma_db",
    )
    assert settings.embedding_models, "At least one embedding model is required"
    assert all(isinstance(model, str) for model in settings.embedding_models)


def test_ensure_directories_creates_directories(tmp_path: Path) -> None:
    # Use Settings directly
    settings = Settings(
        data_dir=tmp_path / "data",
        chroma_db_dir=tmp_path / "chroma_db",
    )

    assert settings.data_dir.exists()
    assert settings.chroma_db_dir.exists()
    assert settings._dirs_ensured


def test_ensure_directories_idempotent(tmp_path: Path) -> None:
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

    # Pre-create only one directory
    data_dir.mkdir(parents=True)
    assert data_dir.exists()
    assert not chroma_db_dir.exists()

    settings = Settings(
        data_dir=data_dir,
        chroma_db_dir=chroma_db_dir,
    )

    assert settings.data_dir.exists()
    assert settings.chroma_db_dir.exists()
    assert settings._dirs_ensured
