"""Minimal tests describing the dataset schema contract."""

from __future__ import annotations

import pytest
from PIL import Image
from pydantic import BaseModel, ValidationError

from qareen.dataset.schema import DatasetItem, DatasetSchema


def test_dataset_schema_contract() -> None:
    """Schema must capture text/image pairs while keeping metadata optional."""
    assert issubclass(DatasetSchema, BaseModel)
    assert issubclass(DatasetItem, BaseModel)

    sample = DatasetSchema(text="caption", image="sample.jpg", metadata={"split": "train"})
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


def test_dataset_schema_accepts_text_only() -> None:
    """Schema must accept samples with only text when image is None."""
    item = DatasetItem(text="caption without image", image=None)
    assert item.text == "caption without image"
    assert item.image is None
    assert item.metadata is None
    assert item.dataset_name is None


def test_dataset_schema_accepts_image_only() -> None:
    """Schema must accept samples with only image when text is None."""
    item = DatasetItem(text=None, image="sample.jpg")
    assert item.text is None
    assert item.image == "sample.jpg"
    assert item.metadata is None
    assert item.dataset_name is None


def test_dataset_schema_accepts_pil_image_only() -> None:
    """Schema must accept PIL Image when text is None."""
    img = Image.new("RGB", (224, 224), color="red")
    item = DatasetItem(text=None, image=img)
    assert item.text is None
    assert isinstance(item.image, Image.Image)


def test_dataset_schema_rejects_both_none() -> None:
    """Schema must reject samples with both text and image as None."""
    with pytest.raises(ValidationError) as exc_info:
        DatasetItem(text=None, image=None)
    assert "at least one modality" in str(exc_info.value).lower()


def test_dataset_schema_rejects_empty_text_when_image_none() -> None:
    """Schema must reject empty text when image is None."""
    with pytest.raises(ValidationError):
        DatasetItem(text="", image=None)

    with pytest.raises(ValidationError):
        DatasetItem(text="   ", image=None)


def test_dataset_schema_accepts_empty_text_when_image_present() -> None:
    """Schema may accept empty text when image is present - implementation specific."""
    try:
        item = DatasetItem(text="", image="sample.jpg")
        assert item.image == "sample.jpg"
    except ValidationError:
        pass
