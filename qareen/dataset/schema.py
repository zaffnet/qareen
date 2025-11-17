from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    text: str = Field(..., min_length=1)
    image: Any
    metadata: dict[str, Any] | None = None
    dataset_name: str | None = None


class DatasetSchema(DatasetItem):
    pass
