"""Pydantic schemas for dataset items.

DEPRECATED: Import DatasetItem from qareen.models instead.
This module is kept for backward compatibility.
"""

from qareen.models import IMAGE_FILE_EXTENSIONS as IMAGE_FILE_EXTENSIONS
from qareen.models import DatasetItem as DatasetItem

__all__ = [
    "DatasetItem",
    "IMAGE_FILE_EXTENSIONS",
]
