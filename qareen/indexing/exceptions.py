from __future__ import annotations


class AlphaNotAvailableError(Exception):
    """Exception raised when an alpha value is not available."""

    def __init__(
        self,
        alpha: float,
        available_alphas: list[float],
        model_id: str,
        dataset_name: str,
        environment: str,
    ):
        """
        Initialize the AlphaNotAvailableError.

        Args:
            alpha: The requested alpha value.
            available_alphas: The list of available alpha values.
            model_id: The ID of the embedding model.
            dataset_name: The name of the dataset.
            environment: The environment.
        """
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
    """Exception raised when a collection name is too long."""

    def __init__(
        self,
        collection_name: str,
        max_length: int,
        suggested_alternatives: list[str] | None = None,
    ):
        """
        Initialize the CollectionNameTooLongError.

        Args:
            collection_name: The collection name.
            max_length: The maximum allowed length.
            suggested_alternatives: A list of suggested alternative names.
        """
        self.collection_name = collection_name
        self.max_length = max_length
        self.suggested_alternatives = suggested_alternatives
        message = f"Collection name '{collection_name}' exceeds max length of {max_length}."
        if suggested_alternatives:
            message += f" Suggested alternatives: {suggested_alternatives}"
        super().__init__(message)


class InvalidCollectionNameError(Exception):
    """Exception raised when a collection name is invalid."""

    def __init__(self, collection_name: str, invalid_characters: list[str]):
        """
        Initialize the InvalidCollectionNameError.

        Args:
            collection_name: The collection name.
            invalid_characters: A list of the invalid characters found.
        """
        self.collection_name = collection_name
        self.invalid_characters = invalid_characters
        super().__init__(
            f"Invalid collection name '{collection_name}'. Invalid characters: {invalid_characters}"
        )
