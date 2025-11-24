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
    """
    Create a Settings instance populated with sensible defaults for tests.
    
    Parameters:
        overrides (object): Keyword overrides for any Settings field; keys are field names and values replace the default values.
    
    Returns:
        Settings: A Settings instance with defaults merged with any provided overrides.
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
    """
    Run garbage collection after each test to release file handles and other resources.
    
    Intended for use as an autouse, function-scoped pytest fixture; yields to the test and invokes gc.collect() after the test completes.
    """
    yield
    gc.collect()