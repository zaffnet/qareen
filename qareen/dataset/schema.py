from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from PIL import Image

class DatasetItem(BaseModel):
    text: str
    image: Any  # Can be a PIL Image or a path
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class DatasetSchema(BaseModel):
    dataset_name: Optional[str] = None
    data: list[DatasetItem]
