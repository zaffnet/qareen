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
            f"Alpha value {alpha:.3f} is not available for "
            f"dataset '{dataset_name}', model '{model_id}', environment '{environment}'. "
            f"Available alpha values: {[f'{a:.3f}' for a in self.available_alphas]}. "
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
        max_length: int = 512,
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


class InvalidAlphaError(ValueError):
    """Raised when alpha value is outside the valid range [0.0, 1.0].

    Attributes:
        alpha: Invalid alpha value
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        message = f"Alpha must be in range [0.0, 1.0], got {alpha}"
        super().__init__(message)


class UnsupportedImageTypeError(ValueError):
    """Raised when image type is not supported.

    Attributes:
        image_type: Type of the unsupported image object
    """

    def __init__(self, image_type: type) -> None:
        self.image_type = image_type
        message = (
            f"Unsupported image type: {image_type}. "
            "Expected PIL.Image, dict with 'bytes', or str path."
        )
        super().__init__(message)


class InvalidEmbeddingError(TypeError):
    """Raised when embedding object does not have required tolist() method.

    Attributes:
        embedding_type: Type of the invalid embedding object
    """

    def __init__(self, embedding_type: type) -> None:
        self.embedding_type = embedding_type
        message = f"Embedding must have tolist() method, got {embedding_type}"
        super().__init__(message)


class CollectionNotFoundError(Exception):
    """Raised when attempting to access a collection that does not exist.

    Attributes:
        collection_name: Name of the missing collection
        dataset_name: Dataset identifier
        model_id: Model identifier
        alpha: Alpha value
        environment: Environment (dev/staging/prod)
    """

    def __init__(
        self,
        collection_name: str,
        dataset_name: str,
        model_id: str,
        alpha: float,
        environment: str,
    ) -> None:
        self.collection_name = collection_name
        self.dataset_name = dataset_name
        self.model_id = model_id
        self.alpha = alpha
        self.environment = environment

        message = (
            f"Collection '{collection_name}' does not exist for "
            f"dataset '{dataset_name}', model '{model_id}', alpha {alpha:.3f}, "
            f"environment '{environment}'. "
            f"Please index the dataset with these parameters before creating a vectorstore."
        )
        super().__init__(message)


class AlphaMismatchError(ValueError):
    """Raised when query alpha does not match collection's indexed alpha.

    Attributes:
        query_alpha: Alpha value used for query
        collection_alpha: Alpha value used to index the collection
    """

    def __init__(self, query_alpha: float, collection_alpha: float) -> None:
        self.query_alpha = query_alpha
        self.collection_alpha = collection_alpha

        message = (
            f"Query alpha {query_alpha:.3f} does not match collection's indexed alpha "
            f"{collection_alpha:.3f}. Query and collection must use the same alpha value "
            f"for semantically correct results. Please use alpha={collection_alpha:.3f} "
            f"in your query or create a vectorstore with alpha={query_alpha:.3f}."
        )
        super().__init__(message)
