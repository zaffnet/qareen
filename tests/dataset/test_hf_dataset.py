import unittest
from unittest.mock import patch

from datasets import Dataset

from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader
from qareen.dataset.schema import DatasetSchema


class TestHuggingFaceDatasetLoader(unittest.TestCase):

    def setUp(self):
        self.dataset_name = "test/dataset"
        self.mock_dataset_dict = {
            "text": ["text1", "text2", "text3"],
            "image": ["image1.png", "image2.png", "image3.png"]
        }
        self.mock_hf_dataset = Dataset.from_dict(self.mock_dataset_dict)

    @patch('datasets.load_dataset')
    def test_load(self, mock_load_dataset):
        mock_load_dataset.return_value = self.mock_hf_dataset
        loader = HuggingFaceDatasetLoader(self.dataset_name)

        dataset_schema = loader.load()
        self.assertIsInstance(dataset_schema, DatasetSchema)
        self.assertEqual(dataset_schema.dataset_name, self.dataset_name)
        self.assertEqual(len(dataset_schema.data), 3)
        self.assertEqual(dataset_schema.data[0].text, "text1")
        self.assertEqual(dataset_schema.data[1].image, "image2.png")

    @patch('datasets.load_dataset')
    def test_load_with_sample_size(self, mock_load_dataset):
        # Create a larger mock dataset
        large_mock_dataset_dict = {
            "text": [f"text{i}" for i in range(10)],
            "image": [f"image{i}.png" for i in range(10)]
        }
        mock_hf_dataset = Dataset.from_dict(large_mock_dataset_dict)
        mock_load_dataset.return_value = mock_hf_dataset

        loader = HuggingFaceDatasetLoader(self.dataset_name, sample_size=5)

        dataset_schema = loader.load()
        self.assertEqual(len(dataset_schema.data), 5)

    def test_get_dataset_name(self):
        loader = HuggingFaceDatasetLoader(self.dataset_name)
        self.assertEqual(loader.get_dataset_name(), self.dataset_name)


if __name__ == '__main__':
    unittest.main()
