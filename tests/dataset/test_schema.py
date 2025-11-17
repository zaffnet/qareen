import os
import unittest

from PIL import Image
from pydantic import ValidationError

from qareen.dataset.schema import DatasetItem, DatasetSchema


class TestSchema(unittest.TestCase):

    def setUp(self):
        # Create a dummy image for testing
        self.image_path = "test_image.png"
        self.image = Image.new('RGB', (100, 100), color = 'red')
        self.image.save(self.image_path)

    def tearDown(self):
        # Remove the dummy image
        if os.path.exists(self.image_path):
            os.remove(self.image_path)

    def test_valid_dataset_item(self):
        item = DatasetItem(text="A red square", image=self.image_path)
        self.assertEqual(item.text, "A red square")
        self.assertEqual(item.image, self.image_path)
        self.assertEqual(item.metadata, {})

        item_with_pil = DatasetItem(text="A red square", image=self.image)
        self.assertEqual(item_with_pil.text, "A red square")
        self.assertIsInstance(item_with_pil.image, Image.Image)

    def test_invalid_dataset_item(self):
        with self.assertRaises(ValidationError):
            DatasetItem(image=self.image_path)  # Missing text
        with self.assertRaises(ValidationError):
            DatasetItem(text="A red square")  # Missing image

    def test_valid_dataset_schema(self):
        items = [
            DatasetItem(text="Item 1", image=self.image_path),
            DatasetItem(text="Item 2", image=self.image_path, metadata={"key": "value"})
        ]
        schema = DatasetSchema(dataset_name="test_dataset", data=items)
        self.assertEqual(schema.dataset_name, "test_dataset")
        self.assertEqual(len(schema.data), 2)
        self.assertEqual(schema.data[0].text, "Item 1")
        self.assertEqual(schema.data[1].metadata, {"key": "value"})

    def test_dataset_schema_defaults(self):
        items = [DatasetItem(text="Item 1", image=self.image_path)]
        schema = DatasetSchema(data=items)
        self.assertIsNone(schema.dataset_name)


if __name__ == '__main__':
    unittest.main()
