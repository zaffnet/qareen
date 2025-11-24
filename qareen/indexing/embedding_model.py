from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

ZERO_NORM_TOLERANCE = 1e-8


class EmbeddingModel(ABC):
    @abstractmethod
    def load_model(self) -> None:
        """
        Load and initialize the underlying embedding model for this instance.
        
        Implementations should prepare any resources or state required for generating embeddings (e.g., loading weights, initializing clients or device placement). This method does not return a value.
        """
        pass

    @abstractmethod
    def embed_text(self, text: str | None) -> np.ndarray | None:
        """
        Produce an embedding vector for the provided text input.
        
        Parameters:
            text (str | None): The text to embed. If `None`, implementations may return `None`.
        
        Returns:
            np.ndarray | None: A 1-D NumPy array representing the text embedding (length equals the model's embedding dimension), or `None` if the input is not embeddable.
        """
        pass

    @abstractmethod
    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        """
        Produce an embedding vector for the provided image input.
        
        Parameters:
            image (PIL.Image.Image | str | Path | None): A PIL Image instance, a filesystem path or string pointing to an image file, or `None`.
        
        Returns:
            numpy.ndarray | None: A NumPy array of shape (embedding_dim,) containing the image embedding, or `None` if no embedding is produced (for example when `image` is `None`).
        """
        pass

    @abstractmethod
    def embed_multimodal(
        self, image: Image.Image | str | Path | None, text: str | None, alpha: float
    ) -> np.ndarray:
        """
        Produce a single multimodal embedding that combines an image embedding and a text embedding using a weighting factor.
        
        Parameters:
            image: A PIL Image, a filesystem path (str or Path) to an image, or `None` to omit the image contribution.
            text: A text string to embed, or `None` to omit the text contribution.
            alpha: A float in [0, 1] that controls the blend: `alpha` scales the image embedding and `1 - alpha` scales the text embedding.
        
        Returns:
            A NumPy array containing the combined embedding vector (shape (embedding_dim,)).
        """
        pass

    @abstractmethod
    def get_model_id(self) -> str:
        """
        Provide a stable identifier for the embedding model.
        
        Returns:
            model_id (str): A string identifying the model (for example, a model name or unique identifier).
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """
        Return the dimensionality of embeddings produced by the model.
        
        Returns:
            int: Number of elements in each embedding vector.
        """
        pass

    @staticmethod
    def normalize_l2(vector: np.ndarray) -> np.ndarray:
        """
        L2-normalizes a numeric NumPy vector.
        
        Parameters:
            vector (np.ndarray): Input numeric vector to normalize. Must be non-empty.
        
        Returns:
            np.ndarray: A vector with the same shape as `vector` scaled to have Euclidean norm 1.
        
        Raises:
            ValueError: If `vector` is empty or its Euclidean norm is less than or equal to ZERO_NORM_TOLERANCE.
        """
        if vector.size == 0:
            raise ValueError("cannot L2-normalize empty vector")
        norm = float(np.linalg.norm(vector))
        if norm <= ZERO_NORM_TOLERANCE:
            raise ValueError("cannot L2-normalize zero vector")
        return vector / norm