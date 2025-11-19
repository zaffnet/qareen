"""Tests for indexing exceptions."""

from __future__ import annotations

from qareen.indexing.exceptions import (
    AlphaNotAvailableError,
    CollectionNameTooLongError,
    CollectionNotFoundError,
    InvalidAlphaError,
    InvalidCollectionNameError,
    InvalidEmbeddingError,
    UnsupportedImageTypeError,
)

TEST_MAX_LENGTH = 512
TEST_LONG_NAME_LENGTH = 700
TEST_SHORT_NAME_LENGTH = 70


def test_alpha_not_available_error():
    error = AlphaNotAvailableError(
        alpha=0.5,
        available_alphas=[0.0, 0.25, 0.75, 1.0],
        model_id="test/model",
        dataset_name="test_dataset",
        environment="dev",
    )

    assert error.alpha == 0.5
    assert error.available_alphas == [0.0, 0.25, 0.75, 1.0]
    assert error.model_id == "test/model"
    assert error.dataset_name == "test_dataset"
    assert error.environment == "dev"
    assert "0.50" in str(error)
    assert "test_dataset" in str(error)
    assert "test/model" in str(error)


def test_collection_name_too_long_error():
    error = CollectionNameTooLongError(
        collection_name="a" * TEST_LONG_NAME_LENGTH,
        max_length=TEST_MAX_LENGTH,
        suggested_alternatives=["short1", "short2"],
    )

    assert error.collection_name == "a" * TEST_LONG_NAME_LENGTH
    assert error.max_length == TEST_MAX_LENGTH
    assert error.suggested_alternatives == ["short1", "short2"]
    assert str(TEST_LONG_NAME_LENGTH) in str(error)
    assert str(TEST_MAX_LENGTH) in str(error)
    assert "short1" in str(error)


def test_collection_name_too_long_error_no_suggestions():
    error = CollectionNameTooLongError(collection_name="a" * TEST_SHORT_NAME_LENGTH)

    assert error.suggested_alternatives == []
    assert "shorter dataset or model name" in str(error)


def test_invalid_collection_name_error():
    error = InvalidCollectionNameError(
        collection_name="test-name!@#",
        invalid_characters={"!", "@", "#"},
    )

    assert error.collection_name == "test-name!@#"
    assert error.invalid_characters == {"!", "@", "#"}
    assert "test-name!@#" in str(error)
    assert "^[a-z0-9_]+$" in str(error)


def test_invalid_alpha_error():
    error = InvalidAlphaError(alpha=1.5)

    assert error.alpha == 1.5
    assert "1.5" in str(error)
    assert "[0.0, 1.0]" in str(error)


def test_unsupported_image_type_error():
    error = UnsupportedImageTypeError(image_type=dict)

    assert error.image_type is dict
    assert "dict" in str(error)
    assert "PIL.Image" in str(error)


def test_invalid_embedding_error():
    error = InvalidEmbeddingError(embedding_type=list)

    assert error.embedding_type is list
    assert "list" in str(error)
    assert "tolist()" in str(error)


def test_collection_not_found_error():
    error = CollectionNotFoundError(
        collection_name="test_collection",
        dataset_name="test_dataset",
        model_id="test/model",
        alpha=0.5,
        environment="prod",
    )

    assert error.collection_name == "test_collection"
    assert error.dataset_name == "test_dataset"
    assert error.model_id == "test/model"
    assert error.alpha == 0.5
    assert error.environment == "prod"
    assert "test_collection" in str(error)
    assert "test_dataset" in str(error)
    assert "0.50" in str(error)
