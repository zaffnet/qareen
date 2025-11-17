"""Dataset schemas expressed with Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetItem(BaseModel):
    """Represents a single text/image pair with optional metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = Field(..., description="Caption or textual description of the sample.")
    image: str | Path | Image.Image = Field(
        ..., description="Image path or a Pillow Image instance associated with the text."
    )
    metadata: Mapping[str, Any] | None = Field(
        default=None, description="Optional metadata payload such as split information."
    )
    dataset_name: str | None = Field(
        default=None, description="Optional dataset identifier for provenance tracking."
    )

    @field_validator("image", mode="before")
    @classmethod
    def _coerce_image(cls, value: Any) -> Any:
        """Coerce pathlib paths to strings to ensure serialization friendliness."""

        if isinstance(value, Path):
            return str(value)
        return value


class DatasetSchema(BaseModel):
    """Schema describing the expected dataset payload shape."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    image: str | Path | Image.Image
    metadata: Mapping[str, Any] | None = None
    dataset_name: str | None = None

    @field_validator("image", mode="before")
    @classmethod
    def _coerce_image(cls, value: Any) -> Any:
        """Align with :class:`DatasetItem` by normalizing pathlib paths."""

        if isinstance(value, Path):
            return str(value)
        return value


__all__ = ["DatasetItem", "DatasetSchema"]
