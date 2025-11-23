"""Consolidated Pydantic models for qareen.

This module contains all Pydantic models used throughout the application,
providing a single source of truth for data validation and settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================================================================
# Dataset Models
# ============================================================================

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
    """Schema for a single dataset item with text and/or image.

    At least one of text or image must be provided. Image paths are validated for
    format/extension at creation time, but file existence is not checked until load.
    """

    INVALID_IMAGE_EXTENSION: ClassVar[str] = "Image path must have valid extension: {path}"
    INVALID_IMAGE_TYPE: ClassVar[str] = "Image must be PIL Image or path string"
    TEXT_EMPTY_ERROR: ClassVar[str] = "Text must be a non-empty string"
    BOTH_NONE_ERROR: ClassVar[str] = "At least one modality (text or image) must be provided"

    text: str | None = None
    image: str | Path | Image.Image | None = None
    metadata: dict[str, Any] | None = Field(default=None)
    dataset_name: str | None = Field(default=None)

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str | None) -> str | None:
        """Validate text is non-empty if provided."""
        if v is not None and not v.strip():
            raise ValueError(cls.TEXT_EMPTY_ERROR)
        if v is not None:
            return v.strip()
        return v

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str | Path | Image.Image | None) -> str | Path | Image.Image | None:
        """Validate image is PIL Image or valid path format if provided.

        Note: Only validates format/type, not file existence. SVG files require extra
        libraries for conversion before PIL/Pillow can load them.
        """
        if v is None:
            return None
        if isinstance(v, (str, Path)):
            path = Path(v)
            if path.suffix.lower() not in IMAGE_FILE_EXTENSIONS:
                raise ValueError(cls.INVALID_IMAGE_EXTENSION.format(path=path))
        elif not isinstance(v, Image.Image):
            raise TypeError(cls.INVALID_IMAGE_TYPE)
        return v

    @model_validator(mode="after")
    def validate_at_least_one_modality(self) -> DatasetItem:
        """Validate that at least one of text or image is provided."""
        if self.text is None and self.image is None:
            raise ValueError(self.BOTH_NONE_ERROR)
        return self

    model_config = {"arbitrary_types_allowed": True}


class Settings(BaseSettings):
    """Configuration settings for qareen.

    Settings can be configured via environment variables (prefixed with QAREEN_),
    config files (.env/qareen.env), or defaults. Precedence: env vars > config file > defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="QAREEN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core configuration
    embedding_models: list[str] = Field(
        description="Embedding model IDs",
    )

    alpha_values: list[float] = Field(
        description="Alpha values for multimodal embedding combination (0.0-1.0)",
    )

    environment: Literal["dev", "staging", "prod"] = Field(
        description="Environment (dev/staging/prod)",
    )

    # Directory configuration
    data_dir: Path = Field(
        description="Directory for dataset storage",
    )

    chroma_db_dir: Path = Field(
        description="Directory for ChromaDB storage",
    )

    # Dataset configuration
    dataset_path: str | None = Field(
        description="Path to dataset (local directory or HuggingFace Hub name)",
    )

    dev_sample_size: int = Field(
        description="Number of samples to use in development environment",
        gt=0,
    )

    max_image_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum image download size in bytes",
        gt=0,
    )

    # Indexing configuration
    batch_size: int = Field(
        description="Batch size for indexing operations",
        gt=0,
    )

    rebuild_collections: bool = Field(
        description="Delete existing collections before indexing",
    )

    # Retrieval configuration
    k_neighbors: int = Field(
        description="Number of similar items to retrieve in queries",
        gt=0,
    )

    # Reproducibility
    random_seed: int = Field(
        description="Random seed for reproducible sampling",
    )

    # Dataset preparation
    dataset_prep_sample_size: int = Field(
        description="Sample size for dataset preparation",
        gt=0,
    )

    prepared_dataset_dir: Path = Field(
        description="Output directory for prepared datasets",
    )

    # Visualization configuration
    viz_output_file: Path = Field(
        description="Output file path for visualization markdown",
    )

    _dirs_ensured: bool = PrivateAttr(default=False)

    @field_validator("alpha_values")
    @classmethod
    def validate_alpha_values(cls, v: list[float]) -> list[float]:
        """Validate alpha values are in [0.0, 1.0] range and deduplicate."""
        if not v:
            raise ValueError("At least one alpha value is required")

        for alpha in v:
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"Alpha value {alpha} must be in range [0.0, 1.0]")

        return sorted(set(v))

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, v: str) -> str:
        """Normalize environment to lowercase."""
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("embedding_models")
    @classmethod
    def validate_models(cls, v: list[str]) -> list[str]:
        """Validate at least one model is provided and deduplicate."""
        if not v:
            raise ValueError("At least one embedding model is required")
        return list(dict.fromkeys(v))

    def model_post_init(self, __context: object) -> None:
        """Initialize after model creation."""
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create filesystem directories if they do not exist.

        Handles race conditions where directories might be created concurrently.
        """
        if self._dirs_ensured:
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_dataset_dir.mkdir(parents=True, exist_ok=True)

        # Ensure viz output directory exists
        self.viz_output_file.parent.mkdir(parents=True, exist_ok=True)

        self._dirs_ensured = True

    def create_embedding_model(self, model_id: str | None = None) -> Any:
        """Create embedding model from configured or provided ID.

        Args:
            model_id: Optional model identifier. If not provided, uses first configured model.

        Returns:
            Instantiated embedding model
        """
        from qareen.indexing.marqo_fashion_model import MarqoFashionSigLIPModel
        from qareen.indexing.siglip_model import SIGLIPEmbeddingModel

        mid = model_id or self.embedding_models[0]
        if mid.lower().startswith("marqo/"):
            return MarqoFashionSigLIPModel(model_id=mid)
        return SIGLIPEmbeddingModel(model_id=mid)

    def create_dataset_loader(self, dataset_path: str | None = None) -> Any:
        """Create dataset loader from configured or provided path.

        Args:
            dataset_path: Optional dataset path. If not provided, uses configured path.

        Returns:
            DatasetLoader instance (HuggingFace or Local)

        Raises:
            ValueError: If no dataset path is configured or provided
        """
        from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader
        from qareen.dataset.local_dataset import LocalDatasetLoader

        path_str = dataset_path or self.dataset_path
        if path_str is None:
            raise ValueError("dataset_path must be set in Settings or provided as argument")

        path = Path(path_str)
        if path.exists():
            return LocalDatasetLoader(dataset_path=path_str)
        return HuggingFaceDatasetLoader(dataset_name=path_str, split="train")
