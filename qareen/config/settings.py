from typing import List, Literal
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    default_embedding_models: List[str] = ["google/siglip-base-patch16-224"]
    data_dir: Path = Path("data/")
    chroma_db_dir: Path = Path("chroma_db/")
    dev_sample_size: int = 1000
    environment: Literal["dev", "staging", "prod"] = "dev"
    alphas: List[float] = [0.5]

settings = Settings()
