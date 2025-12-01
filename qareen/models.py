from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["DatasetItem", "Settings", "IMAGE_FILE_EXTENSIONS"]

IMAGE_FILE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".tiff",
        ".tif",
        ".avif",
        ".heic",
        ".heif",
        ".jfif",
        ".svg",
    }
)


class DatasetItem(BaseModel):
    text: str | None = None
    image: str | Path | Image.Image | None = None
    metadata: dict[str, Any] | None = None
    dataset_name: str | None = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str | None) -> str | None:
        if v is not None:
            if not v.strip():
                raise ValueError("Text must be a non-empty string")
            return v.strip()
        return v

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str | Path | Image.Image | None) -> str | Path | Image.Image | None:
        if v is None:
            return None
        if isinstance(v, (str, Path)):
            if Path(v).suffix.lower() not in IMAGE_FILE_EXTENSIONS:
                raise ValueError(f"Image path must have valid extension: {v}")
        elif not isinstance(v, Image.Image):
            raise TypeError("Image must be PIL Image or path string")
        return v

    @model_validator(mode="after")
    def validate_at_least_one_modality(self) -> DatasetItem:
        if self.text is None and self.image is None:
            raise ValueError("At least one modality (text or image) must be provided")
        return self

    model_config = {"arbitrary_types_allowed": True}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QAREEN_", env_file_encoding="utf-8", extra="ignore"
    )

    embedding_models: list[str] = Field(default=["google/siglip-base-patch16-224"])
    alpha_values: list[float] = Field(default=[0.0, 0.5, 1.0])
    environment: Literal["dev", "staging", "prod"] = Field(default="dev")
    data_dir: Path = Field(default=Path("data"))
    chroma_db_dir: Path = Field(default=Path("chroma_db"))
    dataset_path: str | None = Field(default=None)
    dev_sample_size: int = Field(default=300, gt=0)
    batch_size: int = Field(default=100, gt=0)
    rebuild_collections: bool = Field(default=False)
    k_neighbors: int = Field(default=5, gt=0)
    random_seed: int = Field(default=42)
    dataset_prep_sample_size: int = Field(default=1000, gt=0)
    prepared_dataset_dir: Path = Field(default=Path("data/prepared"))
    viz_output_file: Path = Field(default=Path("data/comparison.md"))
    timeout: float | None = Field(
        default=None,
        gt=0,
        description="Indexing timeout in seconds. None disables timeout.",
    )

    _dirs_ensured: bool = PrivateAttr(default=False)

    @field_validator("alpha_values")
    @classmethod
    def validate_alpha_values(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("At least one alpha value is required")
        for alpha in v:
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"Alpha value {alpha} must be in range [0.0, 1.0]")
        return sorted(set(v))

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v

    @field_validator("embedding_models")
    @classmethod
    def validate_models(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one embedding model is required")
        return list(dict.fromkeys(v))

    def model_post_init(self, __context: object) -> None:
        self.ensure_directories()

    def ensure_directories(self) -> None:
        if self._dirs_ensured:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.viz_output_file.parent.mkdir(parents=True, exist_ok=True)
        self._dirs_ensured = True

    def create_embedding_model(self, model_id: str | None = None) -> Any:
        from qareen.indexing.marqo_fashion_model import MarqoFashionSigLIPModel
        from qareen.indexing.siglip_model import SIGLIPEmbeddingModel

        mid = model_id or self.embedding_models[0]
        if mid.lower().startswith("marqo/"):
            return MarqoFashionSigLIPModel(model_id=mid)
        return SIGLIPEmbeddingModel(model_id=mid)

    def create_dataset_loader(self, dataset_path: str | None = None) -> Any:
        from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader
        from qareen.dataset.local_dataset import LocalDatasetLoader

        path_str = dataset_path or self.dataset_path
        if path_str is None:
            raise ValueError("dataset_path must be set in Settings or provided as argument")
        path = Path(path_str)
        return (
            LocalDatasetLoader(dataset_path=path_str)
            if path.exists()
            else HuggingFaceDatasetLoader(dataset_name=path_str, split="train")
        )
