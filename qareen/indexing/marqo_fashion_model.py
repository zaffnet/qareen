"""Marqo Fashion SIGLIP model implementation using OpenCLIP."""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import open_clip
import torch
from PIL import Image, UnidentifiedImageError

from qareen.indexing.exceptions import InvalidAlphaError
from qareen.indexing.models import EmbeddingModel

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", message=".*timm.*deprecated.*", category=FutureWarning)

logger = logging.getLogger(__name__)


class MarqoFashionSigLIPModel(EmbeddingModel):
    """Marqo Fashion SIGLIP model with special handling for meta tensor issues.

    Attributes:
        model_id: HuggingFace model identifier
        device: Device to run model on (cuda/cpu)
        model: Loaded model instance
        preprocess_val: Preprocessing function for validation transforms
        tokenizer: Tokenizer instance for text encoding

    """

    IMAGE_TYPE_ERROR: str = "Image must be PIL Image or path string"

    def __init__(self, model_id: str = "Marqo/marqo-fashionSigLIP") -> None:
        """Initialize Marqo Fashion SIGLIP model.

        Args:
            model_id: HuggingFace model identifier

        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Any | None = None
        self.preprocess_val: Any | None = None
        self.tokenizer: Any | None = None
        self._cached_embedding_dim: int | None = None

    def load_model(self) -> None:
        """Load Marqo Fashion SIGLIP model using OpenCLIP."""
        model_name = f"hf-hub:{self.model_id}"

        if self.model is None:
            try:
                self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(
                    model_name,
                )
                self.model.eval()
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
                logger.info(
                    "Successfully loaded model: model_id=%s, device=%s",
                    self.model_id,
                    self.device,
                )
            except Exception as e:
                logger.exception("Failed to load model '%s'", self.model_id)
                raise RuntimeError(
                    f"Failed to load Marqo Fashion SIGLIP model '{self.model_id}': {e}",
                ) from e

        if self.tokenizer is None:
            try:
                self.tokenizer = open_clip.get_tokenizer(model_name)
                logger.info("Successfully loaded tokenizer: model_id=%s", self.model_id)
            except Exception as e:
                logger.exception("Failed to load tokenizer for model '%s'", self.model_id)
                raise RuntimeError(
                    f"Failed to load tokenizer for model '{self.model_id}': {e}",
                ) from e

    def embed_text(self, text: str | None) -> np.ndarray | None:
        """Generate L2-normalized text embedding.

        Args:
            text: Input text string or None

        Returns:
            L2-normalized text embedding vector or None if text is None

        """
        if text is None:
            return None

        if self.model is None or self.tokenizer is None:
            self.load_model()

        assert self.tokenizer is not None
        assert self.model is not None

        text_tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    text_features = self.model.encode_text(text_tokens, normalize=True)
            else:
                text_features = self.model.encode_text(text_tokens, normalize=True)

        embedding = text_features[0].cpu().numpy()
        return embedding

    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        """Generate L2-normalized image embedding.

        Args:
            image: PIL Image object, path to image file, or None

        Returns:
            L2-normalized image embedding vector or None if image is None

        """
        if image is None:
            return None

        if self.model is None or self.preprocess_val is None:
            self.load_model()

        if isinstance(image, (str, Path)):
            try:
                with Image.open(image) as im:
                    im.load()
                    image = im.copy()
            except (FileNotFoundError, UnidentifiedImageError) as e:
                raise ValueError(f"{self.IMAGE_TYPE_ERROR}: {e}") from e

        if not isinstance(image, Image.Image):
            raise TypeError(self.IMAGE_TYPE_ERROR)

        assert self.preprocess_val is not None
        assert self.model is not None

        image_input = self.preprocess_val(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    image_features = self.model.encode_image(image_input, normalize=True)
            else:
                image_features = self.model.encode_image(image_input, normalize=True)

        embedding = image_features[0].cpu().numpy()
        return embedding

    def embed_multimodal(
        self,
        image: Image.Image | str | Path | None,
        text: str | None,
        alpha: float,
    ) -> np.ndarray:
        """Generate combined multimodal embedding with alpha weighting.

        Args:
            image: PIL Image object, path to image file, or None
            text: Input text string or None
            alpha: Weight for image embedding (0.0-1.0)

        Returns:
            L2-normalized combined embedding vector

        """
        if not (0.0 <= alpha <= 1.0):
            raise InvalidAlphaError(alpha)

        image_embedding = self.embed_image(image)
        text_embedding = self.embed_text(text)

        if image_embedding is None and text_embedding is None:
            raise ValueError("At least one modality must be present")

        if image_embedding is None:
            return cast(np.ndarray, text_embedding)

        if text_embedding is None:
            return cast(np.ndarray, image_embedding)

        combined = alpha * image_embedding + (1 - alpha) * text_embedding
        return self.normalize_l2(combined)

    def get_model_id(self) -> str:
        """Return normalized model identifier.

        Returns:
            Normalized model identifier

        """
        normalized = self.model_id.lower()
        normalized = re.sub(r"[^a-z0-9_\-/]+", "_", normalized)
        return normalized

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            Embedding dimension as integer

        """
        if self._cached_embedding_dim is not None:
            return self._cached_embedding_dim

        if self.model is None:
            self.load_model()

        try:
            embedding = self.embed_text("dummy")
            if embedding is None:
                raise RuntimeError(
                    f"Failed to determine embedding dimension for model '{self.model_id}'",
                )
            self._cached_embedding_dim = len(embedding)
            return self._cached_embedding_dim
        except Exception as err:
            raise RuntimeError(
                f"Failed to determine embedding dimension for model '{self.model_id}'",
            ) from err
