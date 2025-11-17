"""Settings and configuration for qareen."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ALPHA_REQUIRED_ERR = "At least one alpha value is required"
ALPHA_RANGE_ERR = "Alpha value {alpha} must be in range [0.0, 1.0]"
EMBEDDING_MODEL_REQUIRED_ERR = "At least one embedding model is required"


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

    embedding_models: list[str] = Field(
        default=["google/siglip-base-patch16-224"],
        description="Embedding model IDs",
    )

    alpha_values: list[float] = Field(
        default=[0.5],
        description="Alpha values for multimodal embedding combination (0.0-1.0)",
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

    @field_validator("alpha_values")
    @classmethod
    def validate_alpha_values(cls, v: list[float]) -> list[float]:
        """Validate alpha values are in [0.0, 1.0] range and deduplicate."""
        if not v:
            raise ValueError(ALPHA_REQUIRED_ERR)

        for alpha in v:
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(ALPHA_RANGE_ERR.format(alpha=alpha))

        return sorted(list(set(v)))

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, v: str) -> str:
        """Normalize environment to lowercase."""
        if isinstance(v, str):
            return v.lower()
        return v

    def __init__(self, **kwargs):
        """Initialize Settings with internal state."""
        super().__init__(**kwargs)
        self._dirs_ensured: bool = False

    def ensure_directories(self) -> bool:
        """Create filesystem directories if they do not exist.

        Handles race conditions where directories might be created concurrently.
        Returns True if directories were ensured (newly created or already existed).

        Returns:
            bool: True if directories were successfully ensured, False otherwise.
        """
        if self._dirs_ensured:
            return True

        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
            self._dirs_ensured = True
            return True
        except (FileExistsError, OSError):
            if self.data_dir.exists() and self.chroma_db_dir.exists():
                self._dirs_ensured = True
                return True
            raise

    @field_validator("embedding_models")
    @classmethod
    def validate_models(cls, v: list[str]) -> list[str]:
        """Validate at least one model is provided and deduplicate."""
        if not v:
            raise ValueError(EMBEDDING_MODEL_REQUIRED_ERR)
        return list(dict.fromkeys(v))
