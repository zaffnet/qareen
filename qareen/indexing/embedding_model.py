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
        pass

    @abstractmethod
    def embed_text(self, text: str | None) -> np.ndarray | None:
        pass

    @abstractmethod
    def embed_image(self, image: Image.Image | str | Path | None) -> np.ndarray | None:
        pass

    @abstractmethod
    def embed_multimodal(
        self, image: Image.Image | str | Path | None, text: str | None, alpha: float
    ) -> np.ndarray:
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
            raise ValueError("cannot L2-normalize empty vector")
        norm = float(np.linalg.norm(vector))
        if norm <= ZERO_NORM_TOLERANCE:
            raise ValueError("cannot L2-normalize zero vector")
        return vector / norm
