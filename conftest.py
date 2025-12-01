"""Pytest configuration for qareen tests."""

import gc
import logging
import os

import pytest

# Disable ChromaDB telemetry globally for all tests
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress telemetry logging errors
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


@pytest.fixture(autouse=True)
def cleanup_resources():
    """Cleanup resources after each test to prevent file handle leaks."""
    yield
    gc.collect()
