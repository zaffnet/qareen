from typing import Any

from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    text: str
    image: Any  # Can be a PIL Image or a path
    metadata: dict[str, Any] | None = Field(default_factory=dict)

class DatasetSchema(BaseModel):
    dataset_name: str | None = None
    data: list[DatasetItem]
