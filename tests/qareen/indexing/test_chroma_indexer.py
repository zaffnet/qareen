from __future__ import annotations

import pytest

from qareen.indexing.chroma_indexer import ChromaIndexer
from qareen.indexing.exceptions import InvalidCollectionNameError


def test_get_collection_name_sanitization():
    indexer = ChromaIndexer()
    assert (
        indexer.get_collection_name(
            dataset_name="sqid-shots",
            environment="staging",
            model_id="google-siglip-base-patch16-224",
            alpha=0.5,
        )
        == "staging_sqid-shots_google-siglip-base-patch16-224_alpha0.50"
    )


def test_get_collection_name_invalid_chars():
    indexer = ChromaIndexer()
    with pytest.raises(InvalidCollectionNameError):
        indexer.get_collection_name(
            dataset_name="SQID Shots",
            environment="Staging",
            model_id="google/siglip-base-patch16-224",
            alpha=0.5,
        )
