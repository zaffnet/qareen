"""Tests for Marqo Fashion SIGLIP model."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from qareen.indexing.marqo_fashion_model import MarqoFashionSigLIPModel
from tests.qareen.indexing.test_fixtures import TEST_EMBEDDING_DIM


def test_init():
    model = MarqoFashionSigLIPModel(model_id="Marqo/test-model")
    assert model.model_id == "Marqo/test-model"
    assert model.device in ["cuda", "cpu"]
    assert model.model is None
    assert model.preprocess_val is None
    assert model.tokenizer is None


def test_init_default():
    model = MarqoFashionSigLIPModel()
    assert model.model_id == "Marqo/marqo-fashionSigLIP"


@patch("qareen.indexing.marqo_fashion_model.open_clip")
def test_load_model(mock_open_clip):
    mock_model = Mock()
    mock_preprocess = Mock()
    mock_tokenizer = Mock()

    mock_open_clip.create_model_and_transforms.return_value = (mock_model, None, mock_preprocess)
    mock_open_clip.get_tokenizer.return_value = mock_tokenizer

    model = MarqoFashionSigLIPModel(model_id="Marqo/test-model")
    model.load_model()

    assert model.model == mock_model
    assert model.preprocess_val == mock_preprocess
    assert model.tokenizer == mock_tokenizer
    mock_model.eval.assert_called_once()


@patch("qareen.indexing.marqo_fashion_model.open_clip")
def test_load_model_failure(mock_open_clip):
    mock_open_clip.create_model_and_transforms.side_effect = Exception("Model not found")

    model = MarqoFashionSigLIPModel(model_id="Marqo/test-model")

    with pytest.raises(RuntimeError, match="Failed to load Marqo Fashion SIGLIP model"):
        model.load_model()


def test_embed_text_none():
    model = MarqoFashionSigLIPModel()
    result = model.embed_text(None)
    assert result is None


def test_embed_image_none():
    model = MarqoFashionSigLIPModel()
    result = model.embed_image(None)
    assert result is None


def test_embed_image_invalid_type():
    model = MarqoFashionSigLIPModel()
    model.model = Mock()
    model.preprocess_val = Mock()

    with pytest.raises(TypeError, match="Image must be PIL Image or path string"):
        model.embed_image({"invalid": "type"})


def test_embed_image_invalid_path():
    model = MarqoFashionSigLIPModel()
    model.model = Mock()
    model.preprocess_val = Mock()

    with pytest.raises(ValueError, match="Image must be PIL Image or path string"):
        model.embed_image("/nonexistent/path.jpg")


def test_embed_multimodal_invalid_alpha():
    model = MarqoFashionSigLIPModel()

    with pytest.raises(ValueError, match="Alpha must be in range"):
        model.embed_multimodal(image=None, text="test", alpha=1.5)

    with pytest.raises(ValueError, match="Alpha must be in range"):
        model.embed_multimodal(image=None, text="test", alpha=-0.1)


def test_embed_multimodal_both_none():
    model = MarqoFashionSigLIPModel()

    with pytest.raises(ValueError, match="At least one modality must be present"):
        model.embed_multimodal(image=None, text=None, alpha=0.5)


def test_embed_multimodal_text_only():
    model = MarqoFashionSigLIPModel()
    mock_text_emb = np.array([1.0, 0.0, 0.0])

    with (
        patch.object(model, "embed_text", return_value=mock_text_emb),
        patch.object(model, "embed_image", return_value=None),
    ):
        result = model.embed_multimodal(image=None, text="test", alpha=0.5)
        np.testing.assert_array_equal(result, mock_text_emb)


def test_embed_multimodal_image_only():
    model = MarqoFashionSigLIPModel()
    mock_image_emb = np.array([0.0, 1.0, 0.0])

    with (
        patch.object(model, "embed_text", return_value=None),
        patch.object(model, "embed_image", return_value=mock_image_emb),
    ):
        result = model.embed_multimodal(image=Mock(), text=None, alpha=0.5)
        np.testing.assert_array_equal(result, mock_image_emb)


def test_embed_multimodal_both_modalities():
    model = MarqoFashionSigLIPModel()
    mock_text_emb = np.array([1.0, 0.0, 0.0])
    mock_image_emb = np.array([0.0, 1.0, 0.0])

    with (
        patch.object(model, "embed_text", return_value=mock_text_emb),
        patch.object(model, "embed_image", return_value=mock_image_emb),
        patch.object(model, "normalize_l2") as mock_normalize,
    ):
        mock_normalize.return_value = np.array([0.5, 0.5, 0.0])
        model.embed_multimodal(image=Mock(), text="test", alpha=0.7)

        call_args = mock_normalize.call_args[0][0]
        expected = 0.7 * mock_image_emb + 0.3 * mock_text_emb
        np.testing.assert_array_almost_equal(call_args, expected)


def test_get_model_id():
    model = MarqoFashionSigLIPModel(model_id="Marqo/marqo-fashionSigLIP")
    normalized = model.get_model_id()

    assert normalized == "marqo/marqo-fashionsiglip"
    assert normalized.islower()


def test_get_model_id_special_chars():
    model = MarqoFashionSigLIPModel(model_id="Marqo/Model@v2.1")
    normalized = model.get_model_id()

    assert normalized == "marqo/model_v2_1"


@patch("qareen.indexing.marqo_fashion_model.open_clip")
def test_embedding_dim_caching(mock_open_clip):
    mock_model = Mock()
    mock_open_clip.create_model_and_transforms.return_value = (mock_model, None, Mock())
    mock_open_clip.get_tokenizer.return_value = Mock()

    model = MarqoFashionSigLIPModel()

    with patch.object(model, "embed_text") as mock_embed:
        mock_embed.return_value = np.zeros(TEST_EMBEDDING_DIM)
        # First access - should call embed_text and populate cache
        dim1 = model.embedding_dim
        assert dim1 == TEST_EMBEDDING_DIM
        assert model._cached_embedding_dim == TEST_EMBEDDING_DIM
        mock_embed.assert_called_once_with("dummy")

        # Second access - should use cached value without calling embed_text again
        dim2 = model.embedding_dim
        assert dim2 == TEST_EMBEDDING_DIM
        mock_embed.assert_called_once()


@patch("qareen.indexing.marqo_fashion_model.open_clip")
def test_embedding_dim_failure(mock_open_clip):
    mock_model = Mock()
    mock_open_clip.create_model_and_transforms.return_value = (mock_model, None, Mock())
    mock_open_clip.get_tokenizer.return_value = Mock()

    model = MarqoFashionSigLIPModel()

    with (
        patch.object(model, "embed_text", return_value=None),
        pytest.raises(RuntimeError, match="Failed to determine embedding dimension"),
    ):
        _ = model.embedding_dim
