import unittest
from unittest.mock import MagicMock, patch

from qareen.dataset.schema import DatasetItem, DatasetSchema
from scripts import build_index


class TestBuildIndexScript(unittest.TestCase):

    @patch('scripts.build_index.ChromaIndexer')
    @patch('scripts.build_index.HuggingFaceDatasetLoader')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main(self, mock_parse_args, mock_loader_class, mock_indexer_class):
        # Mock the arguments
        mock_parse_args.return_value = MagicMock(
            dataset_name="test/dataset",
            models=["model1", "model2"],
            alphas=[0.2, 0.8],
            environment="dev",
            sample_size=100
        )

        # Mock the loader instance and its load method
        mock_loader_instance = mock_loader_class.return_value
        mock_loader_instance.load.return_value = DatasetSchema(
            dataset_name="test/dataset",
            data=[DatasetItem(text="a", image="b")]
        )

        # Mock the indexer instance
        mock_indexer_instance = mock_indexer_class.return_value

        # Run the script's main function
        build_index.main()

        # Assert that the loader was called correctly
        mock_loader_class.assert_called_with(dataset_name="test/dataset", sample_size=100)
        mock_loader_instance.load.assert_called_once()

        # Assert that the indexer was called for each model and alpha
        self.assertEqual(mock_indexer_instance.get_collection_name.call_count, 4)
        mock_indexer_instance.get_collection_name.assert_any_call(
            dataset_name="test/dataset", model_id="model1", alpha=0.2, environment="dev"
        )
        mock_indexer_instance.get_collection_name.assert_any_call(
            dataset_name="test/dataset", model_id="model1", alpha=0.8, environment="dev"
        )
        mock_indexer_instance.get_collection_name.assert_any_call(
            dataset_name="test/dataset", model_id="model2", alpha=0.2, environment="dev"
        )
        mock_indexer_instance.get_collection_name.assert_any_call(
            dataset_name="test/dataset", model_id="model2", alpha=0.8, environment="dev"
        )

if __name__ == '__main__':
    unittest.main()
