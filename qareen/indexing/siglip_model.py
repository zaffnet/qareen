"""SIGLIP embedding model implementation."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoProcessor

from qareen.indexing.exceptions import InvalidAlphaError
from qareen.indexing.models import EmbeddingModel

logger = logging.getLogger(__name__)


class SIGLIPEmbeddingModel(EmbeddingModel):
    """SIGLIP model for multimodal embeddings.

    Uses HuggingFace transformers to load and run SIGLIP models.

    Attributes:
        model_id: HuggingFace model identifier
        device: Device to run model on (cuda/cpu)
        model: Loaded model instance
        processor: Loaded processor instance
    """

    IMAGE_TYPE_ERROR: str = "Image must be PIL Image or path string"

    def __init__(self, model_id: str = "google/siglip-base-patch16-224") -> None:
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
                self.model = AutoModel.from_pretrained(self.model_id)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                error_msg = (
                    f"Failed to load SIGLIP model '{self.model_id}' on device '{self.device}': {e}"
                )
                logger.exception(error_msg)
                raise RuntimeError(error_msg) from e

        if self.processor is None:
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id)
            except Exception as e:
                error_msg = f"Failed to load SIGLIP processor for model '{self.model_id}': {e}"
                logger.exception(error_msg)
                raise RuntimeError(error_msg) from e

    def embed_text(self, text: str | None) -> np.ndarray | None:
        """Generate L2-normalized text embedding.

        Args:
            text: Input text string or None

        Returns:
            L2-normalized text embedding vector or None if text is None
        """
        if text is None:
            return None

        if self.model is None or self.processor is None:
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
        """Generate L2-normalized image embedding.

        Args:
            image: PIL Image object, path to image file, or None

        Returns:
            L2-normalized image embedding vector or None if image is None
        """
        if image is None:
            return None

        if self.model is None or self.processor is None:
            self.load_model()

        if isinstance(image, (str, Path)):
            try:
                image = Image.open(image)
            except (FileNotFoundError, UnidentifiedImageError) as e:
                raise TypeError(f"{self.IMAGE_TYPE_ERROR}: {e}") from e

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

        Formula: V_combined = Normalize(alpha * V_image + (1 - alpha) * V_text)
        Both V_image and V_text are L2-normalized before combination.

        Handles missing modalities:
        - If image is None: returns text embedding
        - If text is None: returns image embedding
        - If both are None: raises ValueError

        Args:
            image: PIL Image object, path to image file, or None
            text: Input text string or None
            alpha: Weight for image embedding (0.0-1.0)

        Returns:
            L2-normalized combined embedding vector

        Raises:
            ValueError: If both image and text are None
            InvalidAlphaError: If alpha is not in range [0.0, 1.0]
        """
        if not (0.0 <= alpha <= 1.0):
            raise InvalidAlphaError(alpha)

        image_embedding = self.embed_image(image)
        text_embedding = self.embed_text(text)

        if image_embedding is None and text_embedding is None:
            raise ValueError("At least one modality must be present")

        if image_embedding is None:
            assert text_embedding is not None
            return text_embedding

        if text_embedding is None:
            assert image_embedding is not None
            return image_embedding

        assert image_embedding is not None
        assert text_embedding is not None
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
        if self.model is None:
            self.load_model()

        try:
            config = self.model.config  # type: ignore[union-attr]
            if hasattr(config, "projection_dim") and config.projection_dim is not None:
                return int(config.projection_dim)
            if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
                return int(config.text_config.hidden_size)
        except AttributeError as e:
            config_state = (
                f"config={getattr(self.model, 'config', None)}, "
                f"has_config={hasattr(self.model, 'config')}"
            )
            logger.warning(
                f"Failed to infer embedding_dim from config for model '{self.model_id}'. "
                f"{config_state}. AttributeError: {e}. "
                f"Falling back to sampling with embed_text('dummy')."
            )

        try:
            embedding = self.embed_text("dummy")
            if embedding is None:
                raise RuntimeError(
                    f"Failed to determine embedding dimension for model '{self.model_id}'"
                ) from None
            return len(embedding)
        except Exception:
            raise RuntimeError(
                f"Failed to determine embedding dimension for model '{self.model_id}'"
            ) from None
