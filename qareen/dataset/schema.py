"""Pydantic schemas for dataset items."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from PIL import Image
from pydantic import BaseModel, Field, field_validator, model_validator

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
        ".svg",
    }
)


class DatasetItem(BaseModel):
    """Schema for a single dataset item with text and/or image.

    Attributes:
        text: Text content (caption, description, etc.) - optional
        image: Image (PIL Image object or path to image file) - optional
        metadata: Optional metadata dictionary
        dataset_name: Optional dataset identifier

    Note:
        At least one of text or image must be provided.
        Image paths are validated for format/extension at creation time,
        but file existence is not checked until the image is loaded.
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
        if v is not None and (not v or not v.strip()):
            raise ValueError(cls.TEXT_EMPTY_ERROR)
        return v

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str | Path | Image.Image | None) -> str | Path | Image.Image | None:
        """Validate image is PIL Image or valid path format if provided.

        Note: Only validates format/type, not file existence. File existence
        is checked when the image is actually loaded (e.g., via Image.open()).
        This allows paths to be constructed before files are downloaded or created.
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


DatasetSchema = DatasetItem
