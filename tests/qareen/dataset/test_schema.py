from pydantic import BaseModel
from qareen.dataset.schema import DatasetItem, DatasetSchema

def test_dataset_item_contract() -> None:
    """Items must contain text/image pairs, and may contain arbitrary metadata."""
    assert issubclass(DatasetItem, BaseModel)

    # Text and image are required
    sample = DatasetItem(text="caption", image="sample.jpg")
    assert sample.text and sample.image

    # Metadata is optional and can be any dict
    sample_with_meta = DatasetItem(text="caption", image="sample.jpg", metadata={"split": "train"})
    assert sample_with_meta.metadata


def test_dataset_schema_contract() -> None:
    """Schema must capture a list of dataset items."""
    assert issubclass(DatasetSchema, BaseModel)

    item = DatasetItem(text="caption", image="sample.jpg", metadata={"split": "train"})
    schema = DatasetSchema(data=[item])
    assert schema.data
    assert len(schema.data) == 1
