from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from qareen.dataset.base import DatasetLoader
    from qareen.indexing.embedding_model import EmbeddingModel

__all__ = ["DatasetItem", "IMAGE_FILE_EXTENSIONS", "Settings"]

# Optimistic list of supported extensions. Runtime support depends on installed libraries.
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
    """
    Represents a single dataset item with optional text and image modalities.

    Validation enforces:
    - Text must be non-empty and non-whitespace when provided
    - Image must be a valid file path (with supported extension) or PIL.Image
    - At least one of text or image must be present

    Note: Image extension validation is optimistic and runtime loading may still fail
    for formats not fully supported by the local Pillow installation.
    """

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
    """
    Global configuration for Qareen.

    Fields:
    - embedding_models: List of model IDs (deduplicated). First is default.
    - alpha_values: List of alpha values for RRF (0.0=text, 1.0=image). Sorted and unique.
    - environment: Deployment environment (dev, staging, prod).
    - data_dir: Base directory for data.
    - chroma_db_dir: Directory for ChromaDB persistence.
    - dataset_path: Path to dataset or HF dataset name.
    - dev_sample_size: Sample size for dev environment.
    - batch_size: Batch size for indexing.
    - rebuild_collections: Force rebuild of collections.
    - k_neighbors: Number of neighbors for retrieval.
    - random_seed: Seed for reproducibility.
    - dataset_prep_sample_size: Sample size for dataset preparation.
    - prepared_dataset_dir: Directory for prepared datasets.
    - viz_output_file: Output path for visualization report.
    """

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

    _dirs_ensured: bool = PrivateAttr(default=False)

    @field_validator("alpha_values")
    @classmethod
    def validate_alpha_values(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("At least one alpha value is required")
        for alpha in v:
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"Alpha value {alpha} must be in range [0.0, 1.0]")
        # Note: Silently deduplicates and sorts alpha values
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
        # Note: Silently deduplicates preserving order
        return list(dict.fromkeys(v))

    def model_post_init(self, __context: object) -> None:
        self.ensure_directories()

    def ensure_directories(self) -> None:
        if self._dirs_ensured:
            return

        for path in [
            self.data_dir,
            self.chroma_db_dir,
            self.prepared_dataset_dir,
            self.viz_output_file.parent,
        ]:
            if path.exists() and not path.is_dir():
                raise ValueError(f"Path '{path}' exists but is not a directory")
            path.mkdir(parents=True, exist_ok=True)

        self._dirs_ensured = True

    def create_embedding_model(self, model_id: str | None = None) -> EmbeddingModel:
        from qareen.indexing.marqo_fashion_model import MarqoFashionSigLIPModel
        from qareen.indexing.siglip_model import SIGLIPEmbeddingModel

        mid = model_id if model_id is not None else self.embedding_models[0]
        if mid.lower().startswith("marqo/"):
            return MarqoFashionSigLIPModel(model_id=mid)
        return SIGLIPEmbeddingModel(model_id=mid)

    def create_dataset_loader(self, dataset_path: str | None = None) -> DatasetLoader:
        from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader
        from qareen.dataset.local_dataset import LocalDatasetLoader

        path_str = dataset_path if dataset_path is not None else self.dataset_path
        if path_str is None:
            raise ValueError("dataset_path must be set in Settings or provided as argument")

        path = Path(path_str)
        # Use local loader if path exists and is a directory, otherwise assume HF
        return (
            LocalDatasetLoader(dataset_path=path_str)
            if path.is_dir()
            else HuggingFaceDatasetLoader(dataset_name=path_str, split="train")
        )
