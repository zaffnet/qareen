"""Minimal tests describing the dataset schema contract."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from qareen.dataset.schema import DatasetItem

INVALID_PAYLOADS = (
    pytest.param({"text": "caption"}, id="missing-image"),
    pytest.param({"image": "sample.jpg"}, id="missing-text"),
)


def test_dataset_item_contract() -> None:
    """Schema must capture text/image pairs while keeping metadata optional."""
    assert issubclass(DatasetItem, BaseModel)

    sample = DatasetItem(text="caption", image="sample.jpg", metadata={"split": "train"})
    assert sample.model_dump() == {
        "text": "caption",
        "image": "sample.jpg",
        "metadata": {"split": "train"},
        "dataset_name": None,
    }

    item = DatasetItem(text="caption", image="img.png")
    assert item.model_dump() == {
        "text": "caption",
        "image": "img.png",
        "metadata": None,
        "dataset_name": None,
    }


@pytest.mark.parametrize("payload", INVALID_PAYLOADS)
def test_dataset_item_requires_text_and_image(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DatasetItem(**payload)
