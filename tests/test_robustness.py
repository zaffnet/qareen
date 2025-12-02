import tempfile
from unittest.mock import MagicMock, patch

import pytest

from qareen.indexing.chroma_indexer import ChromaIndexer


def test_chroma_indexer_timeout():
    # Mock dependencies
    dataset_loader = MagicMock()
    dataset_loader.get_dataset_name.return_value = "test_dataset"

    embedding_model = MagicMock()
    embedding_model.get_model_id.return_value = "test_model"

    settings = MagicMock()
    settings.environment = "dev"
    settings.dev_sample_size = 100
    settings.chroma_db_dir = tempfile.mkdtemp()

    indexer = ChromaIndexer(dataset_loader, embedding_model, settings)

    # Mock dataset to be large enough to trigger timeout
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 1000
    # Make select return itself
    mock_dataset.select.return_value = mock_dataset
    # Make slicing return a dict
    mock_dataset.__getitem__.return_value = {"text": ["t"] * 100, "image": ["i"] * 100}

    dataset_loader.load.return_value = mock_dataset

    # Mock embedding model to return valid embedding
    mock_embedding = MagicMock()
    mock_embedding.tolist.return_value = [0.1] * 128
    embedding_model.embed_multimodal.return_value = mock_embedding

    base_time = 1000.0
    time_call_count = [0]
    delta_small = 0.01
    delta_large = 0.1
    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection

    def mock_time():
        time_call_count[0] += 1
        call_index = time_call_count[0]
        # First call: sets start_time in index() method
        if call_index == 1:
            return base_time
        # Calls 2-5: first batch processing (timeout checks and batch timing)
        # Use small delta to stay within 0.05 timeout
        elif call_index <= 5:
            return base_time + delta_small
        # Call 6+: second batch timeout check fails deterministically
        # Exceeds 0.05 timeout threshold
        else:
            return base_time + delta_large

    with (
        patch.object(indexer, "_get_chroma_client", return_value=mock_chroma_client),
        patch("qareen.indexing.chroma_indexer.load_image") as mock_load_image,
        patch("qareen.indexing.chroma_indexer.time.time") as mock_time_func,
    ):
        mock_load_image.return_value = None
        mock_time_func.side_effect = mock_time

        # Test timeout
        with pytest.raises(TimeoutError):
            indexer.index(
                alpha_values=[0.5],
                rebuild=False,
                batch_size=10,
                timeout=0.05,
            )


def test_build_index_empty_models():
    from scripts.build_index import main

    with (
        patch("scripts.build_index.Settings") as mock_settings_cls,
        patch("scripts.build_index.ChromaIndexer") as mock_indexer,
        patch("scripts.build_index.logger") as mock_logger,
    ):
        mock_settings = MagicMock()
        mock_settings.embedding_models = []
        mock_settings.dataset_path = "fake_dataset"
        mock_settings.environment = "dev"
        mock_settings.alpha_values = [0.5]
        mock_settings.batch_size = 32
        mock_settings.rebuild_collections = False
        mock_settings.dev_sample_size = None
        mock_loader = MagicMock()
        mock_loader.load.return_value = []
        mock_loader.validate_schema.return_value = None
        mock_settings.create_dataset_loader.return_value = mock_loader
        mock_settings_cls.return_value = mock_settings

        # Should return 0 and log warning
        result = main()
        assert result == 0
        mock_settings.create_dataset_loader.assert_called_once()
        mock_logger.warning.assert_called_with("⚠️ No embedding models configured. Nothing to do.")
        mock_indexer.assert_not_called()


def test_build_index_model_failure():
    from scripts.build_index import main

    with (
        patch("scripts.build_index.Settings") as mock_settings_cls,
        patch("scripts.build_index.ChromaIndexer") as mock_indexer,
        patch("scripts.build_index.logger") as mock_logger,
    ):
        mock_settings = MagicMock()
        mock_settings.embedding_models = ["model1", "model2"]
        mock_settings.dataset_path = "test_dataset"
        mock_settings.environment = "dev"
        mock_settings.alpha_values = [0.5]
        mock_settings.batch_size = 32
        mock_settings.rebuild_collections = False
        mock_settings.dev_sample_size = None
        mock_settings.timeout = None
        mock_settings_cls.return_value = mock_settings

        mock_dataset_loader = MagicMock()
        mock_dataset_loader.get_dataset_name.return_value = "test_dataset"
        mock_settings.create_dataset_loader.return_value = mock_dataset_loader

        mock_embedding_model = MagicMock()
        mock_embedding_model.get_model_id.return_value = "model2"
        mock_indexer_instance = MagicMock()
        mock_indexer_instance.index.return_value = {0.5: MagicMock()}
        mock_indexer.return_value = mock_indexer_instance

        # Make first model fail
        mock_settings.create_embedding_model.side_effect = [Exception("Boom"), mock_embedding_model]

        # Should return 1 (failure) because model1 failed
        result = main()
        assert result == 1

        # Check that we logged the exception
        mock_logger.exception.assert_called()
        # Check that we logged the error message listing failed models
        mock_logger.error.assert_called()
        error_call_args = mock_logger.error.call_args[0]
        # Check that we processed the second model (create_embedding_model called twice)
        assert mock_settings.create_embedding_model.call_count == 2
        # Assert that index was called once to confirm the second model was processed
        assert mock_indexer_instance.index.call_count == 1
        # Assert that the failed model name equals "model1"
        assert error_call_args[1] == "model1"
