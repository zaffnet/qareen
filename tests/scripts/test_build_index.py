from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts.build_index import main


@patch("scripts.build_index.argparse.ArgumentParser")
def test_build_index_main(mock_argparse):
    mock_args = MagicMock()
    mock_args.dataset_name = "test_dataset"
    mock_args.models = ["test_model"]
    mock_args.alpha_values = [0.5]
    mock_args.environment = "dev"
    mock_args.sample_size = 100
    mock_args.batch_size = 50
    mock_argparse.return_value.parse_args.return_value = mock_args

    with patch("scripts.build_index.HuggingFaceDatasetLoader") as mock_loader, patch(
        "scripts.build_index.SigLIPEmbeddingModel"
    ) as mock_model, patch("scripts.build_index.ChromaIndexer") as mock_indexer:
        main()
        mock_loader.assert_called_with("test_dataset", sample_size=100)
        mock_loader.return_value.load.assert_called_once()
        mock_model.assert_called_with("test_model")
        mock_model.return_value.load_model.assert_called_once()
        mock_indexer.assert_called_once()
        mock_indexer.return_value.get_collection_name.assert_called_with(
            dataset_name="test_dataset",
            environment="dev",
            model_id="test_model",
            alpha=0.5,
        )
