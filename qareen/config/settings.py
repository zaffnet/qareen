from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    default_embedding_models: list[str] = ["google/siglip-base-patch16-224"]
    data_dir: Path = Path("data/")
    chroma_db_dir: Path = Path("chroma_db/")
    dev_sample_size: int = 1000
    environment: Literal["dev", "staging", "prod"] = "dev"
    alphas: list[float] = Field(default_factory=lambda: [0.5])

settings = Settings()
