from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qareen.indexing.chroma_indexer import ChromaIndexer


def test_get_collection_name():
    indexer = ChromaIndexer()
    assert (
        indexer.get_collection_name(
            dataset_name="SQID Shots",
            environment="Staging",
            model_id="google/siglip-base-patch16-224",
            alpha=0.5,
        )
        == "staging_sqid_shots_google_siglip-base-patch16-224_alpha0_50"
    )
