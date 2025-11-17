"""Configuration settings behavior."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

from qareen.config.settings import Settings


def test_settings_defaults_match_plan() -> None:
    assert issubclass(Settings, BaseSettings)

    settings = Settings()
    assert settings.embedding_models, "At least one embedding model is required"
    assert all(isinstance(model, str) for model in settings.embedding_models)

    assert settings.data_dir == Path("data")
    assert settings.chroma_db_dir == Path("chroma_db")
    assert settings.dev_sample_size == 300
    assert settings.environment in {"dev", "staging", "prod"}


def test_ensure_directories_creates_directories(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        chroma_db_dir=tmp_path / "chroma_db",
    )

    assert not settings.data_dir.exists()
    assert not settings.chroma_db_dir.exists()
    assert not settings._dirs_ensured

    result = settings.ensure_directories()

    assert result is True
    assert settings.data_dir.exists()
    assert settings.chroma_db_dir.exists()
    assert settings._dirs_ensured


def test_ensure_directories_idempotent(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        chroma_db_dir=tmp_path / "chroma_db",
    )

    result1 = settings.ensure_directories()
    result2 = settings.ensure_directories()
    result3 = settings.ensure_directories()

    assert result1 is True
    assert result2 is True
    assert result3 is True
    assert settings._dirs_ensured
    assert settings.data_dir.exists()
    assert settings.chroma_db_dir.exists()


def test_ensure_directories_with_existing_directories(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    chroma_db_dir = tmp_path / "chroma_db"
    data_dir.mkdir(parents=True)
    chroma_db_dir.mkdir(parents=True)

    settings = Settings(
        data_dir=data_dir,
        chroma_db_dir=chroma_db_dir,
    )

    assert settings.data_dir.exists()
    assert settings.chroma_db_dir.exists()
    assert not settings._dirs_ensured

    result = settings.ensure_directories()

    assert result is True
    assert settings._dirs_ensured


def test_ensure_directories_with_partial_existing_directories(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    chroma_db_dir = tmp_path / "chroma_db"

    settings = Settings(
        data_dir=data_dir,
        chroma_db_dir=chroma_db_dir,
    )

    data_dir.mkdir(parents=True)

    result = settings.ensure_directories()

    assert result is True
    assert settings.data_dir.exists()
    assert settings.chroma_db_dir.exists()
    assert settings._dirs_ensured
