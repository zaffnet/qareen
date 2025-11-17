"""Application configuration powered by Pydantic settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import EnvSettingsSource


class Settings(BaseSettings):
    """Centralized runtime configuration for qareen."""

    default_embedding_models: list[str] = Field(
        default_factory=lambda: ["google/siglip-base-patch16-224"],
        description="Default embedding models to build indexes for.",
    )
    default_alpha_values: list[float] = Field(
        default_factory=lambda: [0.5],
        description="Mixing weights for text/image embeddings (0-1).",
    )
    data_dir: Path = Field(default=Path("data"), description="Directory where datasets live.")
    chroma_db_dir: Path = Field(
        default=Path("chroma_db"), description="Directory used for Chroma persistence."
    )
    dev_sample_size: int = Field(1000, description="Number of samples to index in dev mode.")
    environment: Literal["dev", "staging", "prod"] = Field(
        default="dev", description="Current deployment environment."
    )

    model_config = SettingsConfigDict(
        env_prefix="QAREEN_",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            LenientEnvSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    @field_validator("default_embedding_models", mode="before")
    @classmethod
    def _coerce_embedding_models(
        cls, value: list[str] | tuple[str, ...] | str | None
    ) -> list[str] | tuple[str, ...] | None:
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            if not parts:
                raise ValueError("At least one embedding model must be configured.")
            return parts
        return value

    @field_validator("default_embedding_models", mode="after")
    @classmethod
    def _validate_embedding_models(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("At least one embedding model must be configured.")
        normalized = [model.strip() for model in values if model and model.strip()]
        if not normalized:
            raise ValueError("At least one embedding model must be configured.")
        return normalized

    @field_validator("default_alpha_values", mode="before")
    @classmethod
    def _coerce_alpha_values(
        cls, value: list[float] | tuple[float, ...] | str | None
    ) -> list[float] | tuple[float, ...] | None:
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            if not parts:
                raise ValueError("At least one alpha value must be configured.")
            return [float(part) for part in parts]
        return value

    @field_validator("default_alpha_values", mode="after")
    @classmethod
    def _validate_alpha_values(cls, values: list[float]) -> list[float]:
        if not values:
            raise ValueError("At least one alpha value must be configured.")
        normalized: list[float] = []
        for alpha in values:
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("Alpha values must be between 0 and 1 inclusive.")
            normalized.append(float(alpha))
        return normalized


__all__ = ["Settings"]


class LenientEnvSettingsSource(EnvSettingsSource):
    """Env source that falls back to raw strings when JSON decoding fails."""

    def decode_complex_value(self, field_name, field, value):  # type: ignore[override]
        try:
            return super().decode_complex_value(field_name, field, value)
        except json.JSONDecodeError:
            return value
