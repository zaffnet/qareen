from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class DatasetLoader(ABC):
    @abstractmethod
    def load(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def validate_schema(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_dataset_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        raise NotImplementedError
