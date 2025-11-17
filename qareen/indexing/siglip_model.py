"""SIGLIP embedding model implementation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from qareen.indexing.models import EmbeddingModel


class SIGLIPEmbeddingModel(EmbeddingModel):
    """SIGLIP model for multimodal embeddings.

    Uses HuggingFace transformers to load and run SIGLIP models.

    Attributes:
        model_id: HuggingFace model identifier
        device: Device to run model on (cuda/cpu)
        model: Loaded model instance
        processor: Loaded processor instance
    """

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
            self.model = AutoModel.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_id)

    def embed_text(self, text: str) -> np.ndarray:
        """Generate L2-normalized text embedding.

        Args:
            text: Input text string

        Returns:
            L2-normalized text embedding vector
        """
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

    def embed_image(self, image: Image.Image | str | Path) -> np.ndarray:
        """Generate L2-normalized image embedding.

        Args:
            image: PIL Image object or path to image file

        Returns:
            L2-normalized image embedding vector
        """
        if self.model is None or self.processor is None:
            self.load_model()

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if not isinstance(image, Image.Image):
            raise TypeError("Image must be PIL Image or path string")

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
        image: Image.Image | str | Path,
        text: str,
        alpha: float,
    ) -> np.ndarray:
        """Generate combined multimodal embedding with alpha weighting.

        Formula: V_combined = Normalize(alpha * V_image + (1 - alpha) * V_text)
        Both V_image and V_text are L2-normalized before combination.

        Args:
            image: PIL Image object or path to image file
            text: Input text string
            alpha: Weight for image embedding (0.0-1.0)

        Returns:
            L2-normalized combined embedding vector
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in range [0.0, 1.0], got {alpha}")

        image_embedding = self.embed_image(image)
        text_embedding = self.embed_text(text)

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
