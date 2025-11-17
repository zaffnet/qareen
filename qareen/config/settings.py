"""Application configuration powered by Pydantic settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized runtime configuration for qareen."""

    default_embedding_models: list[str] = Field(
        default_factory=lambda: ["google/siglip-base-patch16-224"],
        description="Default embedding models to build indexes for.",
    )
    data_dir: Path = Field(default=Path("data"), description="Directory where datasets live.")
    chroma_db_dir: Path = Field(
        default=Path("chroma_db"), description="Directory used for Chroma persistence."
    )
    dev_sample_size: int = Field(1000, description="Number of samples to index in dev mode.")
    environment: Literal["dev", "staging", "prod"] = Field(
        default="dev", description="Current deployment environment."
    )

    model_config = {
        "env_prefix": "QAREEN_",
        "extra": "ignore",
    }


__all__ = ["Settings"]
