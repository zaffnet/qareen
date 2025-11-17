from __future__ import annotations

from typing import List


class AlphaNotAvailableError(Exception):
    def __init__(
        self,
        alpha: float,
        available_alphas: List[float],
        model_id: str,
        dataset_name: str,
        environment: str,
    ):
        self.alpha = alpha
        self.available_alphas = available_alphas
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.environment = environment
        super().__init__(
            f"Alpha {alpha} not available for {model_id} on {dataset_name} in {environment}. "
            f"Available alphas: {available_alphas}"
        )


class CollectionNameTooLongError(Exception):
    def __init__(self, collection_name: str, max_length: int, suggested_alternatives: List[str] | None = None):
        self.collection_name = collection_name
        self.max_length = max_length
        self.suggested_alternatives = suggested_alternatives
        message = f"Collection name '{collection_name}' exceeds max length of {max_length}."
        if suggested_alternatives:
            message += f" Suggested alternatives: {suggested_alternatives}"
        super().__init__(message)


class InvalidCollectionNameError(Exception):
    def __init__(self, collection_name: str, invalid_characters: List[str]):
        self.collection_name = collection_name
        self.invalid_characters = invalid_characters
        super().__init__(
            f"Invalid collection name '{collection_name}'. "
            f"Invalid characters: {invalid_characters}"
        )
