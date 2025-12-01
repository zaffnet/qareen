"""Tests for SIGLIP embedding model."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from qareen.indexing.siglip_model import SIGLIPEmbeddingModel


def test_init():
    model = SIGLIPEmbeddingModel(model_id="google/siglip-base-patch16-224")
    assert model.model_id == "google/siglip-base-patch16-224"
    assert model.device in ["cuda", "cpu"]
    assert model.model is None
    assert model.processor is None


def test_init_default():
    model = SIGLIPEmbeddingModel()
    assert model.model_id == "google/siglip2-base-patch16-512"


@patch("qareen.indexing.siglip_model.AutoModel")
@patch("qareen.indexing.siglip_model.AutoProcessor")
def test_load_model(mock_processor_cls, mock_model_cls):
    mock_model = Mock()
    mock_processor = Mock()
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_processor_cls.from_pretrained.return_value = mock_processor

    model = SIGLIPEmbeddingModel(model_id="test/model")
    model.load_model()

    assert model.model == mock_model
    assert model.processor == mock_processor
    mock_model.to.assert_called_once()
    mock_model.eval.assert_called_once()
    mock_processor_cls.from_pretrained.assert_called_once_with(
        "test/model", trust_remote_code=True, use_fast=True
    )


@patch("qareen.indexing.siglip_model.AutoModel")
def test_load_model_failure(mock_model_cls):
    mock_model_cls.from_pretrained.side_effect = Exception("Model not found")

    model = SIGLIPEmbeddingModel(model_id="test/model")

    with pytest.raises(RuntimeError, match="Failed to load SIGLIP model"):
        model.load_model()


def test_embed_text_none():
    model = SIGLIPEmbeddingModel()
    result = model.embed_text(None)
    assert result is None


def test_embed_image_none():
    model = SIGLIPEmbeddingModel()
    result = model.embed_image(None)
    assert result is None


def test_embed_image_invalid_type():
    model = SIGLIPEmbeddingModel()
    model.model = Mock()
    model.processor = Mock()

    with pytest.raises(TypeError, match="Image must be PIL Image or path string"):
        model.embed_image({"invalid": "type"})


def test_embed_image_invalid_path():
    model = SIGLIPEmbeddingModel()
    model.model = Mock()
    model.processor = Mock()

    with pytest.raises(ValueError, match="Image must be PIL Image or path string"):
        model.embed_image("/nonexistent/path.jpg")


def test_embed_multimodal_invalid_alpha():
    model = SIGLIPEmbeddingModel()

    with pytest.raises(ValueError, match="Alpha must be in range"):
        model.embed_multimodal(image=None, text="test", alpha=1.5)

    with pytest.raises(ValueError, match="Alpha must be in range"):
        model.embed_multimodal(image=None, text="test", alpha=-0.1)


def test_embed_multimodal_both_none():
    model = SIGLIPEmbeddingModel()

    with pytest.raises(ValueError, match="At least one modality must be present"):
        model.embed_multimodal(image=None, text=None, alpha=0.5)


def test_embed_multimodal_text_only():
    model = SIGLIPEmbeddingModel()
    mock_text_emb = np.array([1.0, 0.0, 0.0])

    with (
        patch.object(model, "embed_text", return_value=mock_text_emb),
        patch.object(model, "embed_image", return_value=None),
    ):
        result = model.embed_multimodal(image=None, text="test", alpha=0.5)
        np.testing.assert_array_equal(result, mock_text_emb)


def test_embed_multimodal_image_only():
    model = SIGLIPEmbeddingModel()
    mock_image_emb = np.array([0.0, 1.0, 0.0])

    with (
        patch.object(model, "embed_text", return_value=None),
        patch.object(model, "embed_image", return_value=mock_image_emb),
    ):
        result = model.embed_multimodal(image=Mock(), text=None, alpha=0.5)
        np.testing.assert_array_equal(result, mock_image_emb)


def test_embed_multimodal_both_modalities():
    model = SIGLIPEmbeddingModel()
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
    model = SIGLIPEmbeddingModel(model_id="Google/SigLIP-Base-Patch16")
    normalized = model.get_model_id()

    assert normalized == "google/siglip-base-patch16"
    assert normalized.islower()
    assert "/" in normalized


def test_get_model_id_special_chars():
    model = SIGLIPEmbeddingModel(model_id="Test/Model@v2.1")
    normalized = model.get_model_id()

    assert normalized == "test/model_v2_1"


@patch("qareen.indexing.siglip_model.AutoModel")
@patch("qareen.indexing.siglip_model.AutoProcessor")
def test_embedding_dim_from_config(mock_processor_cls, mock_model_cls):
    mock_model = Mock()
    mock_model.config.projection_dim = 63
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_processor_cls.from_pretrained.return_value = Mock()

    model = SIGLIPEmbeddingModel()
    assert model.embedding_dim == 63


@patch("qareen.indexing.siglip_model.AutoModel")
@patch("qareen.indexing.siglip_model.AutoProcessor")
def test_embedding_dim_from_text_config(mock_processor_cls, mock_model_cls):
    mock_model = Mock()
    mock_model.config.projection_dim = None
    mock_model.config.text_config.hidden_size = 768
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_processor_cls.from_pretrained.return_value = Mock()

    model = SIGLIPEmbeddingModel()
    assert model.embedding_dim == 768


@patch("qareen.indexing.siglip_model.AutoModel")
@patch("qareen.indexing.siglip_model.AutoProcessor")
def test_embedding_dim_fallback_to_embed(mock_processor_cls, mock_model_cls):
    mock_model = Mock()
    del mock_model.config
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_processor_cls.from_pretrained.return_value = Mock()

    model = SIGLIPEmbeddingModel()

    with patch.object(model, "embed_text") as mock_embed:
        mock_embed.return_value = np.zeros(384)
        assert model.embedding_dim == 384


@patch("qareen.indexing.siglip_model.AutoModel")
@patch("qareen.indexing.siglip_model.AutoProcessor")
def test_embedding_dim_failure(mock_processor_cls, mock_model_cls):
    mock_model = Mock()
    del mock_model.config
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_processor_cls.from_pretrained.return_value = Mock()

    model = SIGLIPEmbeddingModel()

    with (
        patch.object(model, "embed_text", return_value=None),
        pytest.raises(RuntimeError, match="Failed to determine embedding dimension"),
    ):
        _ = model.embedding_dim
