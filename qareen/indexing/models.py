from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoProcessor, ProcessorMixin, SiglipModel


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def load_model(self) -> Any:
        """Load the embedding model."""
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a string of text."""
        raise NotImplementedError

    @abstractmethod
    def embed_image(self, image: Image.Image | str) -> np.ndarray:
        """Embed an image."""
        raise NotImplementedError

    @abstractmethod
    def embed_multimodal(self, image: Any, text: str, alpha: float) -> np.ndarray:
        """Embed a multimodal combination of text and image."""
        raise NotImplementedError

    @abstractmethod
    def get_model_id(self) -> str:
        """Get the model ID."""
        raise NotImplementedError


class SigLIPEmbeddingModel(EmbeddingModel):
    """An embedding model for SigLIP."""

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        device: str = "cpu",
    ):
        """
        Initialize the SigLIPEmbeddingModel.

        Args:
            model_id: The ID of the SigLIP model to use.
            device: The device to use for inference.
        """
        self.model_id = model_id
        self.device = device
        self.model: SiglipModel | None = None
        self.processor: ProcessorMixin | None = None

    def load_model(self) -> Any:
        """Load the embedding model."""
        if self.model is None:
            self.model = SiglipModel.from_pretrained(self.model_id).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        return self.model

    def embed_text(self, text: str) -> np.ndarray:
        if self.processor is None or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        inputs = self.processor(text=[text], return_tensors="pt")
        text_features = self.model.get_text_features(**inputs)
        embedding = text_features.detach().cpu().numpy()
        return embedding / np.linalg.norm(embedding)

    def embed_image(self, image: Image.Image | str) -> np.ndarray:
        if self.processor is None or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Image file not found: {image}") from e
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        embedding = image_features.detach().cpu().numpy()
        return embedding / np.linalg.norm(embedding)

    def embed_multimodal(self, image: Any, text: str, alpha: float) -> np.ndarray:
        """Embed a multimodal combination of text and image."""
        image_embedding = self.embed_image(image)
        text_embedding = self.embed_text(text)
        combined = (alpha * image_embedding) + ((1 - alpha) * text_embedding)
        norm = np.linalg.norm(combined)
        if norm > 0:
            return combined / norm
        return combined

    def get_model_id(self) -> str:
        """Get the model ID."""
        return self.model_id
