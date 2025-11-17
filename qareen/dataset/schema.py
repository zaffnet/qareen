from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    """A single item in a dataset."""

    text: str = Field(..., min_length=1)
    image: Any
    metadata: dict[str, Any] | None = None
    dataset_name: str | None = None
