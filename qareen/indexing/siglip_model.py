from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoProcessor

from qareen.indexing.embedding_model import EmbeddingModel


class SIGLIPEmbeddingModel(EmbeddingModel):
    def __init__(self, model_id: str = "google/siglip2-base-patch16-512") -> None:
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: AutoModel | None = None
        self.processor: AutoProcessor | None = None

    def load_model(self) -> None:
        if self.model is None:
            try:
                self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SIGLIP model '{self.model_id}' on device '{self.device}': {e}"
                ) from e

        if self.processor is None:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id, trust_remote_code=True, use_fast=True
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SIGLIP processor for model '{self.model_id}': {e}"
                ) from e

    def embed_text(self, text: str | None) -> np.ndarray | None:
        if text is None:
            return None
        self.load_model()
        assert self.processor is not None
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)  # type: ignore[union-attr]
        return self.normalize_l2(outputs[0].cpu().numpy())

    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        if image is None:
            return None
        self.load_model()

        if isinstance(image, (str, Path)):
            try:
                with Image.open(image) as im:
                    im.load()
                    image = im.copy()
            except (FileNotFoundError, UnidentifiedImageError) as e:
                raise ValueError(f"Image must be PIL Image or path string: {e}") from e

        if not isinstance(image, Image.Image):
            raise TypeError("Image must be PIL Image or path string")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)  # type: ignore[misc]
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)  # type: ignore[union-attr]
        return self.normalize_l2(outputs[0].cpu().numpy())

    def embed_multimodal(
        self, image: Image.Image | str | Path | None, text: str | None, alpha: float
    ) -> np.ndarray:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in range [0.0, 1.0], got {alpha}")

        image_emb = self.embed_image(image)
        text_emb = self.embed_text(text)

        if image_emb is None and text_emb is None:
            raise ValueError("At least one modality must be present")
        if image_emb is None:
            return cast(np.ndarray, text_emb)
        if text_emb is None:
            return cast(np.ndarray, image_emb)

        return self.normalize_l2(alpha * image_emb + (1 - alpha) * text_emb)

    def get_model_id(self) -> str:
        normalized = re.sub(r"[^a-z0-9_\-/]+", "_", self.model_id.lower())
        return normalized

    @property
    def embedding_dim(self) -> int:
        if self.model is None:
            self.load_model()

        try:
            config = self.model.config  # type: ignore[union-attr]
            if hasattr(config, "projection_dim") and config.projection_dim is not None:
                return int(config.projection_dim)
            if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
                return int(config.text_config.hidden_size)
        except AttributeError:
            # Fall back to inferring dimension from dummy embedding
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
