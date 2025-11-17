"""CLI tests for scripts.download_sqid."""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from scripts.download_sqid import main


def test_download_sqid_sample_size_logs_correct_row_count(caplog: pytest.LogCaptureFixture) -> None:
    """Test that --sample-size logs the correct number of rows after sampling."""
    caplog.set_level(logging.INFO)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "data"
        output_dir.mkdir(parents=True)

        mock_dataset = Dataset.from_dict(
            {
                "text": [f"text_{i}" for i in range(100)],
                "image": [f"image_{i}" for i in range(100)],
            }
        )

        with patch("scripts.download_sqid.HuggingFaceDatasetLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_loader.get_dataset_name.return_value = "test_dataset"
            mock_loader.split = "train"
            mock_loader.validate_schema.return_value = True
            mock_loader_class.return_value = mock_loader

            with patch("scripts.download_sqid.Settings") as mock_settings:
                mock_settings.return_value.data_dir = output_dir
                mock_settings.return_value.ensure_directories.return_value = None

                with patch.object(sys, "argv", ["download_sqid.py", "--sample-size", "5"]):
                    result = main()

                assert result == 0
                mock_loader.load.assert_called()
                mock_loader.validate_schema.assert_called()
                mock_loader_class.assert_called()
                mock_settings.assert_called()

        log_messages = [record.message for record in caplog.records]
        info_log = [msg for msg in log_messages if "Dataset info:" in msg]
        assert len(info_log) == 1

        assert "5 rows" in info_log[0]
