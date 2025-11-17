from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.download_sqid import main


@patch("scripts.download_sqid.argparse.ArgumentParser")
def test_download_sqid_main(mock_argparse):
    mock_args = MagicMock()
    mock_args.dataset_name = "test_dataset"
    mock_args.output_dir = Path("test_output")
    mock_args.validate = True
    mock_args.sample_size = 100
    mock_argparse.return_value.parse_args.return_value = mock_args

    with patch("scripts.download_sqid.HuggingFaceDatasetLoader") as mock_loader:
        main()
        mock_loader.assert_called_with("test_dataset", sample_size=100)
        mock_loader.return_value.load.assert_called_once()
        mock_loader.return_value.validate_schema.assert_called_once()
