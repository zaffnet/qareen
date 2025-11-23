"""SIGLIP embedding model implementation."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoProcessor

from qareen.indexing.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SIGLIPEmbeddingModel(EmbeddingModel):
    """SIGLIP model for multimodal embeddings.

    Attributes:
        model_id: HuggingFace model identifier
        device: Device to run model on (cuda/cpu)
        model: Loaded model instance
        processor: Loaded processor instance
    """

    IMAGE_TYPE_ERROR: str = "Image must be PIL Image or path string"

    def __init__(self, model_id: str = "google/siglip2-base-patch16-512") -> None:
        """Initialize SIGLIP model.

        Args:
            model_id: HuggingFace model identifier
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: AutoModel | None = None
        self.processor: AutoProcessor | None = None

    def load_model(self) -> None:
        """Load HuggingFace SIGLIP model and processor."""
        if self.model is None:
            try:
                self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                error_msg = (
                    f"Failed to load SIGLIP model '{self.model_id}' on device '{self.device}': {e}"
                )
                raise RuntimeError(error_msg) from e

        if self.processor is None:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id, trust_remote_code=True, use_fast=True
                )
            except Exception as e:
                error_msg = f"Failed to load SIGLIP processor for model '{self.model_id}': {e}"
                raise RuntimeError(error_msg) from e

    def embed_text(self, text: str | None) -> np.ndarray | None:
        """Generate L2-normalized text embedding."""
        if text is None:
            return None

        self.load_model()

        inputs = self.processor(  # type: ignore[misc]
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)  # type: ignore[union-attr]

        embedding = outputs[0].cpu().numpy()
        return self.normalize_l2(embedding)

    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        """Generate L2-normalized image embedding."""
        if image is None:
            return None

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

        inputs = self.processor(  # type: ignore[misc]
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)  # type: ignore[union-attr]

        embedding = outputs[0].cpu().numpy()
        return self.normalize_l2(embedding)

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
            raise ValueError(f"Alpha must be in range [0.0, 1.0], got {alpha}")

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
        """Return normalized model identifier."""
        normalized = self.model_id.lower()
        normalized = re.sub(r"[^a-z0-9_\-/]+", "_", normalized)
        return normalized

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension.

        Note: Accessing this property may trigger model loading.
        """
        if self.model is None:
            self.load_model()

        try:
            config = self.model.config  # type: ignore[union-attr]
            if hasattr(config, "projection_dim") and config.projection_dim is not None:
                return int(config.projection_dim)
            if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
                return int(config.text_config.hidden_size)
        except AttributeError:
            # Attribute(s) not found in config; fallback to computing embedding dimension from dummy input below.
            pass

        try:
            embedding = self.embed_text("dummy")
            if embedding is None:
                raise RuntimeError(
                    f"Failed to determine embedding dimension for model '{self.model_id}'"
                )
            return len(embedding)
        except Exception as err:
            raise RuntimeError(
                f"Failed to determine embedding dimension for model '{self.model_id}'"
            ) from err
