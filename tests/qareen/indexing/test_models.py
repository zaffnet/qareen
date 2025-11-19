"""Tests for embedding model base class."""

from __future__ import annotations

import numpy as np
import pytest

from qareen.indexing.models import EmbeddingModel


def test_normalize_l2_valid_vector():
    vector = np.array([3.0, 4.0])
    normalized = EmbeddingModel.normalize_l2(vector)

    expected = np.array([0.6, 0.8])
    np.testing.assert_array_almost_equal(normalized, expected)
    assert np.isclose(np.linalg.norm(normalized), 1.0)


def test_normalize_l2_already_normalized():
    vector = np.array([0.6, 0.8])
    normalized = EmbeddingModel.normalize_l2(vector)

    np.testing.assert_array_almost_equal(normalized, vector)
    assert np.isclose(np.linalg.norm(normalized), 1.0)


def test_normalize_l2_zero_vector():
    vector = np.array([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match=EmbeddingModel.ZERO_VECTOR_ERROR):
        EmbeddingModel.normalize_l2(vector)


def test_normalize_l2_near_zero_vector():
    vector = np.array([1e-10, 1e-10, 1e-10])

    with pytest.raises(ValueError, match=EmbeddingModel.ZERO_VECTOR_ERROR):
        EmbeddingModel.normalize_l2(vector)


def test_normalize_l2_multidimensional():
    vector = np.array([1.0, 2.0, 2.0])
    normalized = EmbeddingModel.normalize_l2(vector)

    assert np.isclose(np.linalg.norm(normalized), 1.0)
    expected = vector / np.linalg.norm(vector)
    np.testing.assert_array_almost_equal(normalized, expected)
