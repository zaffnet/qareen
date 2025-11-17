"""Custom exceptions for indexing operations."""

from __future__ import annotations


class AlphaNotAvailableError(Exception):
    """Raised when querying with an alpha value that hasn't been indexed.

    Attributes:
        alpha: Requested alpha value
        available_alphas: List of available alpha values
        model_id: Model identifier
        dataset_name: Dataset name
        environment: Environment (dev/staging/prod)
    """

    def __init__(
        self,
        alpha: float,
        available_alphas: list[float],
        model_id: str,
        dataset_name: str,
        environment: str,
    ) -> None:
        self.alpha = alpha
        self.available_alphas = sorted(available_alphas)
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.environment = environment

        message = (
            f"Alpha value {alpha:.2f} is not available for "
            f"dataset '{dataset_name}', model '{model_id}', environment '{environment}'. "
            f"Available alpha values: {[f'{a:.2f}' for a in self.available_alphas]}. "
            f"Please re-index with the desired alpha value."
        )
        super().__init__(message)


class CollectionNameTooLongError(Exception):
    """Raised when collection name exceeds maximum length.

    Attributes:
        collection_name: Generated collection name
        max_length: Maximum allowed length
        suggested_alternatives: List of suggested shorter names
    """

    def __init__(
        self,
        collection_name: str,
        max_length: int = 63,
        suggested_alternatives: list[str] | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.max_length = max_length
        self.suggested_alternatives = suggested_alternatives or []

        message = (
            f"Collection name '{collection_name}' exceeds maximum length of {max_length} "
            f"characters (current: {len(collection_name)}). "
        )

        if self.suggested_alternatives:
            message += f"Suggested alternatives: {', '.join(self.suggested_alternatives)}"
        else:
            message += "Please use a shorter dataset or model name."

        super().__init__(message)


class InvalidCollectionNameError(Exception):
    """Raised when collection name contains invalid characters.

    Attributes:
        collection_name: Generated collection name
        invalid_characters: Set of invalid characters found
    """

    def __init__(
        self,
        collection_name: str,
        invalid_characters: set[str],
    ) -> None:
        self.collection_name = collection_name
        self.invalid_characters = invalid_characters

        message = (
            f"Collection name '{collection_name}' contains invalid characters: "
            f"{sorted(invalid_characters)}. "
            f"Collection names must match pattern ^[a-z0-9_]+$"
        )
        super().__init__(message)
