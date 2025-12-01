"""Pytest configuration for qareen tests."""

import gc
import logging
import os
from pathlib import Path

import pytest

from qareen.models import Settings

# Disable ChromaDB telemetry globally for all tests
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress telemetry logging errors
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def create_test_settings(**overrides: object) -> Settings:
    """Create Settings instance with all required fields for testing.

    Args:
        **overrides: Field values to override defaults

    Returns:
        Settings instance with all required fields set
    """
    defaults = {
        "embedding_models": ["google/siglip-base-patch16-224"],
        "alpha_values": [0.5],
        "environment": "dev",
        "data_dir": Path("data"),
        "chroma_db_dir": Path("chroma_db"),
        "dataset_path": None,
        "dev_sample_size": 300,
        "batch_size": 100,
        "rebuild_collections": False,
        "k_neighbors": 5,
        "random_seed": 42,
        "dataset_prep_sample_size": 1000,
        "prepared_dataset_dir": Path("data/prepared"),
        "viz_output_file": Path("data/comparison.md"),
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture(autouse=True)
def cleanup_resources():
    """Cleanup resources after each test to prevent file handle leaks."""
    yield
    gc.collect()
