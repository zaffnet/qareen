"""Configuration settings behavior."""

from __future__ import annotations

from pathlib import Path

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


def test_settings_deduplicate_alpha_values_preserving_order() -> None:
    settings = Settings(default_alpha_values=[0.2, 0.2, 0.5, 0.2])
    assert settings.default_alpha_values == [0.2, 0.5]
