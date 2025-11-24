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
        """
        Initialize the embedding model wrapper, selecting device and storing configuration.
        
        Parameters:
            model_id (str): Identifier of the pretrained model to use (default: "google/siglip2-base-patch16-512").
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: AutoModel | None = None
        self.processor: AutoProcessor | None = None

    def load_model(self) -> None:
        """
        Ensure the SIGLIP model and processor for this instance are loaded and ready for inference.
        
        Loads the pretrained model onto the instance's selected device and sets it to evaluation mode, and loads the corresponding processor. If either component is already loaded, it is left unchanged.
        
        Raises:
            RuntimeError: If loading the model or the processor fails; the exception message includes the model identifier and the underlying error.
        """
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
        """
        Compute an L2-normalized embedding vector for the given text.
        
        Parameters:
            text (str | None): The input text to embed. If `None`, the function returns `None`.
        
        Returns:
            np.ndarray | None: A 1-D NumPy array containing the L2-normalized embedding for the input text, or `None` if `text` is `None`.
        """
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
        """
        Compute an L2-normalized embedding for the given image.
        
        Parameters:
            image (PIL.Image.Image | str | Path | None): A PIL Image or a filesystem path (string or Path) pointing to an image file. If `None`, the function returns `None`.
        
        Returns:
            np.ndarray | None: L2-normalized embedding vector for the image as a NumPy array, or `None` when `image` is `None`.
        
        Raises:
            ValueError: If a provided path does not exist or the file cannot be identified as an image.
            TypeError: If `image` is not a PIL Image and not a path string/Path.
        """
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
        """
        Compute a fused embedding from image and text using a weighted combination.
        
        Parameters:
            image (PIL.Image.Image | str | Path | None): Image input, a PIL Image or a filesystem path to an image. Pass None to omit the image modality.
            text (str | None): Text input. Pass None to omit the text modality.
            alpha (float): Weight for the image embedding in the fusion; must be between 0.0 and 1.0 inclusive. The text weight is (1 - alpha).
        
        Returns:
            np.ndarray: L2-normalized embedding vector computed as alpha * image_embedding + (1 - alpha) * text_embedding, or the single-modality embedding when only one modality is provided.
        
        Raises:
            ValueError: If alpha is outside [0.0, 1.0], or if both image and text are None.
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
        Convert the instance's model_id to a normalized identifier safe for filenames and keys.
        
        Returns:
            str: The model_id lowercased with any character not in [a-z0-9_-/] replaced by an underscore.
        """
        normalized = re.sub(r"[^a-z0-9_\-/]+", "_", self.model_id.lower())
        return normalized

    @property
    def embedding_dim(self) -> int:
        """
        Determine the dimensionality of embeddings produced by the loaded model.
        
        Attempts to read the dimension from the model configuration (e.g., `projection_dim` or `text_config.hidden_size`); if that fails, computes a dummy text embedding and returns its length.
        
        Returns:
            int: The embedding vector length.
        
        Raises:
            RuntimeError: If the embedding dimension cannot be determined.
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