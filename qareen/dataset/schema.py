from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    text: str = Field(..., min_length=1)
    image: Any
    metadata: Optional[Dict[str, Any]] = None
    dataset_name: Optional[str] = None


class DatasetSchema(DatasetItem):
    pass
