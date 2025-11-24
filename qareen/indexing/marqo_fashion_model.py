from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import open_clip
import torch
from PIL import Image, UnidentifiedImageError

from qareen.indexing.embedding_model import EmbeddingModel

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", message=".*timm.*deprecated.*", category=FutureWarning)


class MarqoFashionSigLIPModel(EmbeddingModel):
    def __init__(self, model_id: str = "Marqo/marqo-fashionSigLIP") -> None:
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Any | None = None
        self.preprocess_val: Any | None = None
        self.tokenizer: Any | None = None
        self._cached_embedding_dim: int | None = None

    def load_model(self) -> None:
        model_name = f"hf-hub:{self.model_id}"

        if self.model is None:
            try:
                self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(
                    model_name
                )
                self.model.eval()
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Marqo Fashion SIGLIP model '{self.model_id}': {e}"
                ) from e

        if self.tokenizer is None:
            try:
                self.tokenizer = open_clip.get_tokenizer(model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load tokenizer for model '{self.model_id}': {e}"
                ) from e

    def embed_text(self, text: str | None) -> np.ndarray | None:
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
        return text_features[0].cpu().numpy()

    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
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
                raise ValueError(f"Image must be PIL Image or path string: {e}") from e

        if not isinstance(image, Image.Image):
            raise TypeError("Image must be PIL Image or path string")

        assert self.preprocess_val is not None
        assert self.model is not None

        image_input = self.preprocess_val(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    image_features = self.model.encode_image(image_input, normalize=True)
            else:
                image_features = self.model.encode_image(image_input, normalize=True)
        return image_features[0].cpu().numpy()

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
        return re.sub(r"[^a-z0-9_\-/]+", "_", self.model_id.lower())

    @property
    def embedding_dim(self) -> int:
        if self._cached_embedding_dim is not None:
            return self._cached_embedding_dim
        if self.model is None:
            self.load_model()
        try:
            embedding = self.embed_text("dummy")
            if embedding is None:
                raise RuntimeError(
                    f"Failed to determine embedding dimension for model '{self.model_id}'"
                )
            self._cached_embedding_dim = len(embedding)
            return self._cached_embedding_dim
        except Exception as err:
            raise RuntimeError(
                f"Failed to determine embedding dimension for model '{self.model_id}'"
            ) from err
