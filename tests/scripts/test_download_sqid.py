import unittest
from unittest.mock import patch, MagicMock
from scripts import download_sqid
from qareen.dataset.schema import DatasetSchema, DatasetItem

class TestDownloadSqidScript(unittest.TestCase):

    @patch('scripts.download_sqid.HuggingFaceDatasetLoader')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main(self, mock_parse_args, mock_loader_class):
        # Mock the arguments
        mock_parse_args.return_value = MagicMock(dataset_name="test/dataset")

        # Mock the loader instance and its load method
        mock_loader_instance = mock_loader_class.return_value
        mock_loader_instance.load.return_value = DatasetSchema(
            dataset_name="test/dataset",
            data=[DatasetItem(text="a", image="b"), DatasetItem(text="c", image="d")]
        )

        # Run the script's main function
        download_sqid.main()

        # Assert that the loader was called with the correct dataset name
        mock_loader_class.assert_called_with(dataset_name="test/dataset")
        # Assert that the load method was called
        mock_loader_instance.load.assert_called_once()

if __name__ == '__main__':
    unittest.main()
