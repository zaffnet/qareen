from __future__ import annotations

from pathlib import Path
from typing import List, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="qareen_")

    default_embedding_models: List[str] = Field(
        default_factory=lambda: ["google/siglip-base-patch16-224"]
    )
    default_alpha_values: List[float] = Field(default_factory=lambda: [0.5])
    data_dir: Path = Field(default="data/")
    chroma_db_dir: Path = Field(default="chroma_db/")
    dev_sample_size: int = 1000
    environment: Literal["dev", "staging", "prod"] = "dev"
