"""Minimal tests describing the dataset schema contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from pydantic import BaseModel, ValidationError

from qareen.models import DatasetItem


def test_schema_pydantic_model() -> None:
    """Test that DatasetItem is a pydantic model."""
    assert issubclass(DatasetItem, BaseModel)


@pytest.fixture
def sample_img_path(tmp_path: Path) -> Path:
    """Fixture to create a sample image file."""
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (224, 224), color="red").save(path)
    return path


def test_dataset_sample_creation(sample_img_path: Path) -> None:
    """Test creation of a dataset sample with both text and image."""
    sample = DatasetItem(text="caption", image=str(sample_img_path), metadata={"split": "train"})
    assert sample.model_dump() == {
        "text": "caption",
        "image": str(sample_img_path),
        "metadata": {"split": "train"},
        "dataset_name": None,
    }


def test_dataset_schema_accepts_text_only() -> None:
    """Schema must accept samples with only text when image is None."""
    item = DatasetItem(text="caption without image", image=None)
    assert item.text == "caption without image"
    assert item.image is None
    assert item.metadata is None
    assert item.dataset_name is None


def test_dataset_schema_accepts_image_only(tmp_path: Path) -> None:
    """Schema must accept samples with only image when text is None."""
    sample_img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (224, 224), color="red").save(sample_img_path)

    item = DatasetItem(text=None, image=str(sample_img_path))
    assert item.text is None
    assert item.image == str(sample_img_path)
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


def test_dataset_schema_rejects_empty_text_when_image_present(tmp_path: Path) -> None:
    """Schema must reject empty text when image is present."""
    sample_img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (224, 224), color="red").save(sample_img_path)

    with pytest.raises(ValidationError):
        DatasetItem(text="", image=str(sample_img_path))


def test_dataset_schema_accepts_nonexistent_path_with_valid_extension(tmp_path: Path) -> None:
    """Schema must accept paths with valid extensions even if file doesn't exist yet."""
    nonexistent_path = tmp_path / "future_image.jpg"
    assert not nonexistent_path.exists()

    item = DatasetItem(text="caption", image=str(nonexistent_path))
    assert item.image == str(nonexistent_path)
