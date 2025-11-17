from abc import ABC, abstractmethod
from typing import Any


class EmbeddingModel(ABC):

    @abstractmethod
    def load_model(self) -> Any:
        """Loads the embedding model."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generates text embeddings."""
        pass

    @abstractmethod
    def embed_image(self, image: Any) -> list[float]:
        """Generates image embeddings."""
        pass

    @abstractmethod
    def embed_multimodal(self, text: str, image: Any) -> list[float]:
        """Handles multimodal embedding."""
        pass
