from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_prefix="qareen_")

    default_embedding_models: list[str] = Field(
        default_factory=lambda: ["google/siglip-base-patch16-224"]
    )
    default_alpha_values: list[float] = Field(default_factory=lambda: [0.5])
    data_dir: Path = Field(default=Path("data/"))
    chroma_db_dir: Path = Field(default=Path("chroma_db/"))
    dev_sample_size: int = 1000
    environment: Literal["dev", "staging", "prod"] = "dev"
