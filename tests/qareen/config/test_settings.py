"""Configuration settings behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_settings import BaseSettings

from qareen.config.settings import Settings


def test_settings_defaults_match_plan() -> None:
    assert issubclass(Settings, BaseSettings)

    settings = Settings()
    assert settings.default_embedding_models, "At least one default embedding model is required"
    assert all(isinstance(model, str) for model in settings.default_embedding_models)
    assert settings.default_alpha_values == [0.5]

    assert settings.data_dir == Path("data")
    assert settings.chroma_db_dir == Path("chroma_db")
    assert settings.dev_sample_size == 1000
    assert settings.environment in {"dev", "staging", "prod"}


def test_settings_parses_list_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "QAREEN_DEFAULT_EMBEDDING_MODELS",
        "google/siglip-base-patch16-224, openai/clip-vit-large-patch14",
    )
    monkeypatch.setenv("QAREEN_DEFAULT_ALPHA_VALUES", "0.2, 0.8")

    settings = Settings()
    assert settings.default_embedding_models == [
        "google/siglip-base-patch16-224",
        "openai/clip-vit-large-patch14",
    ]
    assert settings.default_alpha_values == [0.2, 0.8]


def test_settings_rejects_invalid_alpha_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QAREEN_DEFAULT_ALPHA_VALUES", "1.5, -0.1")

    with pytest.raises(ValueError):
        Settings()
