from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image
from transformers import SiglipImageProcessor, SiglipModel


class EmbeddingModel(ABC):
    @abstractmethod
    def load_model(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_image(self, image: Image.Image | str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_multimodal(self, image: Any, text: str, alpha: float) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_model_id(self) -> str:
        raise NotImplementedError


class SigLIPEmbeddingModel(EmbeddingModel):
    def __init__(self, model_id: str = "google/siglip-base-patch16-224"):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load_model(self) -> Any:
        if self.model is None:
            self.model = SiglipModel.from_pretrained(self.model_id)
            self.processor = SiglipImageProcessor.from_pretrained(self.model_id)
        return self.model

    def embed_text(self, text: str) -> np.ndarray:
        assert self.processor is not None
        assert self.model is not None
        inputs = self.processor(text=[text], return_tensors="pt")
        text_features = self.model.get_text_features(**inputs)
        return text_features.detach().numpy()

    def embed_image(self, image: Image.Image | str) -> np.ndarray:
        assert self.processor is not None
        assert self.model is not None
        if isinstance(image, str):
            image = Image.open(image)
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)
        return image_features.detach().numpy()

    def embed_multimodal(self, image: Any, text: str, alpha: float) -> np.ndarray:
        image_embedding = self.embed_image(image)
        text_embedding = self.embed_text(text)
        combined = (alpha * image_embedding) + ((1 - alpha) * text_embedding)
        return combined / np.linalg.norm(combined)

    def get_model_id(self) -> str:
        return self.model_id
