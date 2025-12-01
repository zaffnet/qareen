from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

ZERO_NORM_TOLERANCE = 1e-8
_ERR_EMPTY_VECTOR = "cannot L2-normalize empty vector"
_ERR_INVALID_VALUES = "cannot L2-normalize vector containing NaN or Inf"
_ERR_ZERO_VECTOR = "cannot L2-normalize zero vector"


class EmbeddingModel(ABC):
    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def embed_text(self, text: str | None) -> np.ndarray | None:
        pass

    @abstractmethod
    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        pass

    def embed_multimodal(
        self, image: Image.Image | str | Path | None, text: str | None, alpha: float
    ) -> np.ndarray:
        """Embeds text and image modalities with a weighted combination.

        Args:
            image: Image or path to image.
            text: Text to embed.
            alpha: Weight for image modality (0.0 to 1.0).

        Returns:
            Normalized embedding vector.

        Raises:
            ValueError: If alpha is not in [0.0, 1.0] or if neither modality is provided.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in range [0.0, 1.0], got {alpha}")

        if image is None and text is None:
            raise ValueError("At least one modality must be present")

        return self._embed_multimodal_impl(image, text, alpha)

    @abstractmethod
    def _embed_multimodal_impl(
        self, image: Image.Image | str | Path | None, text: str | None, alpha: float
    ) -> np.ndarray:
        """Implementation of multimodal embedding."""
        pass

    @abstractmethod
    def get_model_id(self) -> str:
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @staticmethod
    def normalize_l2(vector: np.ndarray) -> np.ndarray:
        if vector.size == 0:
            raise ValueError(_ERR_EMPTY_VECTOR)
        if not np.isfinite(vector).all():
            raise ValueError(_ERR_INVALID_VALUES)
        norm = np.linalg.norm(vector)
        if norm <= ZERO_NORM_TOLERANCE:
            raise ValueError(_ERR_ZERO_VECTOR)
        return vector / norm
