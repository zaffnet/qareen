"""Settings and configuration for qareen."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for qareen.

    Settings can be configured via environment variables (prefixed with QAREEN_),
    config files (.env/qareen.env), or defaults.

    Precedence: env vars > config file > defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="QAREEN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    default_embedding_models: list[str] = Field(
        default=["google/siglip-base-patch16-224"],
        description="Default embedding model IDs",
    )

    default_alpha_values: list[float] = Field(
        default=[0.5],
        description="Default alpha values for multimodal embedding combination (0.0-1.0)",
    )

    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for dataset storage",
    )

    chroma_db_dir: Path = Field(
        default=Path("chroma_db"),
        description="Directory for ChromaDB storage",
    )

    dev_sample_size: int = Field(
        default=1000,
        description="Number of samples to use in development environment",
        gt=0,
    )

    environment: Literal["dev", "staging", "prod"] = Field(
        default="dev",
        description="Environment (dev/staging/prod)",
    )

    @field_validator("default_alpha_values")
    @classmethod
    def validate_alpha_values(cls, v: list[float]) -> list[float]:
        """Validate alpha values are in [0.0, 1.0] range and deduplicate."""
        if not v:
            raise ValueError("At least one alpha value is required")

        for alpha in v:
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"Alpha value {alpha} must be in range [0.0, 1.0]")

        return sorted(list(set(v)))

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, v: str) -> str:
        """Normalize environment to lowercase."""
        if isinstance(v, str):
            return v.lower()
        return v

    @model_validator(mode="after")
    def create_directories(self) -> Settings:
        """Ensure directories exist after model instantiation."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        return self

    @field_validator("default_embedding_models")
    @classmethod
    def validate_models(cls, v: list[str]) -> list[str]:
        """Validate at least one model is provided and deduplicate."""
        if not v:
            raise ValueError("At least one default embedding model is required")
        return list(dict.fromkeys(v))
