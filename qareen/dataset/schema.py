from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from PIL import Image

from pydantic import ConfigDict

class DatasetItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: str
    image: Union[str, Image.Image]  # Can be a PIL Image or a path
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class DatasetSchema(BaseModel):
    dataset_name: Optional[str] = None
    data: list[DatasetItem]
