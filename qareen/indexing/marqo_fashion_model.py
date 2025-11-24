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
        """
        Create a MarqoFashionSigLIPModel instance and initialize runtime defaults and placeholders.
        
        Parameters:
            model_id (str): Identifier of the model to load (e.g., "Marqo/marqo-fashionSigLIP"). This value is stored and used when loading the model.
        
        Description:
            Sets the device to "cuda" if a CUDA GPU is available, otherwise "cpu". Initializes placeholders for the loaded model, image preprocessing transform, tokenizer, and a cached embedding dimension.
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Any | None = None
        self.preprocess_val: Any | None = None
        self.tokenizer: Any | None = None
        self._cached_embedding_dim: int | None = None

    def load_model(self) -> None:
        """
        Ensure the instance has a loaded model, image preprocessing transform, and tokenizer, and move the model to the configured device if applicable.
        
        If the model is not already loaded, load the model and image preprocessing transform for this instance's model_id, set the model to evaluation mode, and move it to the configured device when appropriate. If the tokenizer is not already loaded, obtain and set the tokenizer.
        
        Raises:
            RuntimeError: If loading the model, preprocessing transform, or tokenizer fails.
        """
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
        """
        Compute a vector embedding for the provided text using the loaded model.
        
        Parameters:
            text (str | None): Input text to embed; if `None`, no embedding is computed.
        
        Returns:
            np.ndarray | None: `None` if `text` is `None`, otherwise a 1-D NumPy array containing the text embedding.
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
        return text_features[0].cpu().numpy()

    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        """
        Compute a normalized image embedding from a PIL image or image file path.
        
        Parameters:
            image (PIL.Image.Image | str | pathlib.Path | None): A PIL Image object, a filesystem path or path string to an image, or None.
        
        Returns:
            numpy.ndarray | None: A 1-D NumPy array containing the image embedding on success, or `None` if `image` is `None`.
        
        Raises:
            ValueError: If the provided path does not exist or the file cannot be identified as an image.
            TypeError: If `image` is not a PIL Image instance nor a path string/Path.
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
        """
        Compute a fused multimodal embedding from image and/or text using a weighted combination.
        
        Parameters:
            image: An image (PIL Image), a filesystem path, or None. If a path is provided the image is loaded; if None the image modality is omitted.
            text: A string containing the text input, or None to omit the text modality.
            alpha: Weight for the image embedding in the fusion; must be between 0.0 and 1.0 inclusive. The combined embedding is alpha * image + (1 - alpha) * text.
        
        Returns:
            np.ndarray: An L2-normalized embedding vector representing the fused (or single-modality) embedding.
        
        Raises:
            ValueError: If alpha is not in [0.0, 1.0].
            ValueError: If both image and text are None (at least one modality must be present).
        """
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
        """
        Produce a sanitized model identifier suitable for use in file paths and keys.
        
        Returns:
            str: The instance's `model_id` lowercased, with any character other than letters a–z, digits 0–9, underscore `_`, hyphen `-`, or slash `/` replaced by an underscore.
        """
        return re.sub(r"[^a-z0-9_\-/]+", "_", self.model_id.lower())

    @property
    def embedding_dim(self) -> int:
        """
        Return the model's embedding vector dimensionality, caching the result for subsequent calls.
        
        Returns:
            embedding_dim (int): Number of elements in a single embedding produced by the model.
        
        Raises:
            RuntimeError: If the embedding dimensionality cannot be determined.
        """
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