from __future__ import annotations

import pytest

from qareen.indexing.exceptions import (
    AlphaNotAvailableError,
    CollectionNameTooLongError,
    InvalidCollectionNameError,
)


def test_alpha_not_available_error():
    with pytest.raises(AlphaNotAvailableError, match="Alpha 0.5 not available"):
        raise AlphaNotAvailableError(0.5, [0.0, 1.0], "model", "dataset", "dev")


def test_collection_name_too_long_error():
    with pytest.raises(CollectionNameTooLongError, match="exceeds max length of 63"):
        raise CollectionNameTooLongError("a" * 64, 63)


def test_invalid_collection_name_error():
    with pytest.raises(InvalidCollectionNameError, match=r"Invalid characters: \['@'\]"):
        raise InvalidCollectionNameError("test@collection", ["@"])
