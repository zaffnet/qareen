from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
        """
        Trim leading and trailing whitespace from text and ensure it is not empty.
        
        Parameters:
            v (str | None): Input text to validate; may be None.
        
        Returns:
            str | None: The trimmed string if provided, otherwise None.
        
        Raises:
            ValueError: If `v` is a string that becomes empty after trimming.
        """
        if v is not None:
            if not v.strip():
                raise ValueError("Text must be a non-empty string")
            return v.strip()
        return v

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str | Path | Image.Image | None) -> str | Path | Image.Image | None:
        """
        Validate that the provided image value is either a supported file path or a PIL Image.
        
        Parameters:
            v (str | Path | PIL.Image.Image | None): Image input to validate. Accepted values are:
                - None
                - a filesystem path or path-like string with a supported image extension
                - a PIL Image instance
        
        Returns:
            str | Path | PIL.Image.Image | None: The original `v` if it is valid, or `None` when `v` is `None`.
        
        Raises:
            ValueError: If `v` is a path/string whose file extension is not in IMAGE_FILE_EXTENSIONS.
            TypeError: If `v` is neither a path/string nor a PIL Image instance.
        """
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
        """
        Ensure the instance has at least one of the text or image modalities set.
        
        Raises:
            ValueError: If both `text` and `image` are None.
        
        Returns:
            DatasetItem: The validated model instance (`self`).
        """
        if self.text is None and self.image is None:
            raise ValueError("At least one modality (text or image) must be provided")
        return self

    model_config = {"arbitrary_types_allowed": True}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QAREEN_", env_file_encoding="utf-8", extra="ignore"
    )

    embedding_models: list[str] = Field()
    alpha_values: list[float] = Field()
    environment: Literal["dev", "staging", "prod"] = Field()
    data_dir: Path = Field()
    chroma_db_dir: Path = Field()
    dataset_path: str | None = Field(default=None)
    dev_sample_size: int = Field(gt=0)
    max_image_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    batch_size: int = Field(gt=0)
    rebuild_collections: bool = Field()
    k_neighbors: int = Field(gt=0)
    random_seed: int = Field()
    dataset_prep_sample_size: int = Field(gt=0)
    prepared_dataset_dir: Path = Field()
    viz_output_file: Path = Field()

    _dirs_ensured: bool = PrivateAttr(default=False)

    @field_validator("alpha_values")
    @classmethod
    def validate_alpha_values(cls, v: list[float]) -> list[float]:
        """
        Validate and normalize a list of alpha values used for interpolation or weighting.
        
        Parameters:
            v (list[float]): List of alpha values expected to be in the range 0.0 to 1.0 inclusive.
        
        Returns:
            list[float]: A sorted list of unique alpha values.
        
        Raises:
            ValueError: If `v` is empty or any alpha is outside the range 0.0 to 1.0.
        """
        if not v:
            raise ValueError("At least one alpha value is required")
        for alpha in v:
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"Alpha value {alpha} must be in range [0.0, 1.0]")
        return sorted(set(v))

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, v: str) -> str:
        """
        Normalize an environment value to lowercase when it is a string.
        
        Parameters:
            v: The environment value to normalize; if not a string it is returned unchanged.
        
        Returns:
            The lowercased string when `v` is a `str`, otherwise the original value.
        """
        return v.lower() if isinstance(v, str) else v

    @field_validator("embedding_models")
    @classmethod
    def validate_models(cls, v: list[str]) -> list[str]:
        """
        Ensure the embedding model list is non-empty and return it with duplicates removed while preserving order.
        
        Parameters:
            v (list[str]): Sequence of embedding model identifiers.
        
        Returns:
            list[str]: The provided model identifiers with duplicates removed in their original order.
        
        Raises:
            ValueError: If `v` is empty.
        """
        if not v:
            raise ValueError("At least one embedding model is required")
        return list(dict.fromkeys(v))

    def model_post_init(self, __context: object) -> None:
        """
        Perform post-initialization tasks for the Settings model by ensuring required directories exist.
        """
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """
        Ensure required filesystem directories exist for this Settings instance.
        
        Creates data_dir, chroma_db_dir, prepared_dataset_dir, and the parent directory of viz_output_file if they do not already exist. This operation is idempotent and will be skipped after it has run once for the instance.
        """
        if self._dirs_ensured:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.viz_output_file.parent.mkdir(parents=True, exist_ok=True)
        self._dirs_ensured = True

    def create_embedding_model(self, model_id: str | None = None) -> Any:
        """
        Create an embedding model instance based on a model identifier.
        
        Parameters:
            model_id (str | None): Optional model identifier to instantiate; when omitted the first entry from `self.embedding_models` is used.
        
        Returns:
            Any: An embedding model instance — `MarqoFashionSigLIPModel` if the model id starts with "marqo/", otherwise `SIGLIPEmbeddingModel`.
        """
        from qareen.indexing.marqo_fashion_model import MarqoFashionSigLIPModel
        from qareen.indexing.siglip_model import SIGLIPEmbeddingModel

        mid = model_id or self.embedding_models[0]
        if mid.lower().startswith("marqo/"):
            return MarqoFashionSigLIPModel(model_id=mid)
        return SIGLIPEmbeddingModel(model_id=mid)

    def create_dataset_loader(self, dataset_path: str | None = None) -> Any:
        """
        Create a dataset loader for a local dataset path or a Hugging Face dataset name.
        
        If `dataset_path` (or the `Settings.dataset_path` fallback) points to an existing local path, returns a LocalDatasetLoader for that path; otherwise returns a HuggingFaceDatasetLoader for the given dataset name using the "train" split.
        
        Parameters:
            dataset_path (str | None): Optional local path or Hugging Face dataset identifier; if omitted, `self.dataset_path` is used.
        
        Returns:
            A dataset loader instance: `LocalDatasetLoader` when the path exists, otherwise `HuggingFaceDatasetLoader`.
        
        Raises:
            ValueError: If neither `dataset_path` nor `self.dataset_path` is provided.
        """
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