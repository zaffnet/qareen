from pydantic import BaseModel

from qareen.dataset.schema import DatasetItem, DatasetSchema


def test_dataset_item_contract() -> None:
    """Items must contain text/image pairs, and may contain arbitrary metadata."""
    assert issubclass(DatasetItem, BaseModel)

    # Text and image are required
    sample = DatasetItem(text="caption", image="sample.jpg")
    assert sample.text == "caption"
    assert sample.image == "sample.jpg"
    assert sample.metadata == {}  # Verify default

    # Metadata is optional and can be any dict
    sample_with_meta = DatasetItem(text="caption", image="sample.jpg", metadata={"split": "train"})
    assert sample_with_meta.metadata == {"split": "train"}


def test_dataset_schema_contract() -> None:
    """Schema must capture a list of dataset items."""
    assert issubclass(DatasetSchema, BaseModel)

    item = DatasetItem(text="caption", image="sample.jpg", metadata={"split": "train"})
    schema = DatasetSchema(data=[item])
    assert schema.data
    assert len(schema.data) == 1
    assert schema.dataset_name is None  # Verify default
    assert schema.data[0].text == "caption"

    # Test with dataset_name provided
    named_schema = DatasetSchema(dataset_name="test_dataset", data=[item])
    assert named_schema.dataset_name == "test_dataset"
