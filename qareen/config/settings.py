"""Application configuration powered by Pydantic settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


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

    model_config = {
        "env_prefix": "QAREEN_",
        "extra": "ignore",
    }

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
