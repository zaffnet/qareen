"""Pydantic schemas for dataset items."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field, field_validator


class DatasetItem(BaseModel):
    """Schema for a single dataset item with text and image.

    Attributes:
        text: Text content (caption, description, etc.)
        image: Image (PIL Image object or path to image file)
        metadata: Optional metadata dictionary
        dataset_name: Optional dataset identifier
    """

    text: str
    image: str | Path | Image.Image
    metadata: dict[str, Any] | None = Field(default=None)
    dataset_name: str | None = Field(default=None)

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate text is non-empty."""
        if not v or not v.strip():
            raise ValueError("Text must be a non-empty string")
        return v

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str | Path | Image.Image) -> str | Path | Image.Image:
        """Validate image is PIL Image or valid path format."""
        if isinstance(v, (str, Path)):
            path = Path(v)
            if path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".bmp",
                ".webp",
                ".tiff",
                ".tif",
                ".svg",
            }:
                raise ValueError(f"Image path must have valid extension: {path}")
        elif not isinstance(v, Image.Image):
            raise TypeError("Image must be PIL Image or path string")
        return v

    model_config = {"arbitrary_types_allowed": True}


DatasetSchema = DatasetItem
