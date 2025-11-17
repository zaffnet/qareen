"""Dataset loader contracts."""

from __future__ import annotations

from abc import ABC

from qareen.dataset.base import DatasetLoader
from qareen.dataset.hf_dataset import HuggingFaceDatasetLoader

REQUIRED_METHODS = frozenset({"load", "validate_schema", "get_dataset_name", "get_dataset_info"})


def test_dataset_loader_contract_and_hf_impl() -> None:
    assert issubclass(DatasetLoader, ABC)
    assert getattr(DatasetLoader, "__abstractmethods__", set()) >= REQUIRED_METHODS

    assert issubclass(HuggingFaceDatasetLoader, DatasetLoader)
    missing = [
        method
        for method in REQUIRED_METHODS
        if not hasattr(HuggingFaceDatasetLoader, method)
    ]
    assert not missing, f"HuggingFaceDatasetLoader must implement: {missing}"
