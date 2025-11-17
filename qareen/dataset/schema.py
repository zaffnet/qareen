from typing import Any

from pydantic import BaseModel, Field


from pydantic import ConfigDict

class DatasetItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: str
    image: Any  # Can be a PIL Image or a path
    metadata: dict[str, Any] | None = Field(default_factory=dict)

class DatasetSchema(BaseModel):
    dataset_name: str | None = None
    data: list[DatasetItem]
