from __future__ import annotations

import hashlib
import re


def get_collection_name(
    dataset_name: str, model_id: str, alpha: float | None = None, environment: str = "dev"
) -> str:
    """
    Constructs a sanitized collection name from environment, dataset name, model id, and an optional alpha suffix.
    
    Sanitizes inputs to lowercase alphanumeric/underscore segments, collapses repeated underscores, and trims leading/trailing underscores. Produces a name of the form "<env>_<dataset>_<model>" with an optional "_a{alpha:.3f}" suffix. If the dataset and model parts would exceed the maximum allowed length (63 characters) when combined with the environment and suffix, the function truncates the parts and appends a deterministic short hash suffix to keep the final name within limits.
    
    Parameters:
        dataset_name (str): Source dataset name; must be a non-empty string.
        model_id (str): Model identifier; must be a non-empty string.
        alpha (float | None): Optional numeric suffix formatted as `_a{value:.3f}` when provided.
        environment (str): One of "dev", "staging", or "prod" (case-insensitive); defaults to "dev".
    
    Returns:
        str: The constructed, sanitized collection name.
    
    Raises:
        ValueError: If `dataset_name` or `model_id` is empty, if `environment` is not one of the allowed values, or if the final collection name exceeds 63 characters.
    """
    dataset_name = dataset_name.strip()
    model_id = model_id.strip()
    env = environment.strip().lower()

    if not dataset_name:
        raise ValueError("dataset_name must be a non-empty string")
    if not model_id:
        raise ValueError("model_id must be a non-empty string")
    if env not in ("dev", "staging", "prod"):
        raise ValueError(
            f"environment must be one of 'dev', 'staging', or 'prod', got '{environment}'"
        )

    def sanitize(part: str) -> str:
        """
        Sanitize a string into a lowercase, underscore-separated identifier.
        
        Parameters:
        	part (str): Input string to normalize.
        
        Returns:
        	sanitized (str): The input converted to lowercase, with any character that is not a lowercase letter, digit, or underscore replaced by an underscore, consecutive underscores collapsed into one, and any leading or trailing underscores removed.
        """
        return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", part.lower())).strip("_")

    env_part, dataset_part, model_part = sanitize(env), sanitize(dataset_name), sanitize(model_id)
    alpha_suffix = f"_a{alpha:.3f}" if alpha is not None else ""
    max_length = 63
    available_length = max_length - len(env_part) - len(alpha_suffix) - 2

    if len(dataset_part) + len(model_part) <= available_length:
        base_name = f"{env_part}_{dataset_part}_{model_part}"
    else:
        hash_hex = hashlib.sha256(f"{dataset_part}_{model_part}".encode()).hexdigest()[:8]
        hash_suffix = f"_h{hash_hex}"
        available_for_parts = available_length - 10
        half_available = available_for_parts // 2
        dataset_truncated = dataset_part[:half_available]
        model_truncated = model_part[: available_for_parts - len(dataset_truncated)]
        base_name = f"{env_part}_{dataset_truncated}_{model_truncated}{hash_suffix}"

    name = f"{base_name}{alpha_suffix}"
    if len(name) > max_length:
        raise ValueError(
            f"Collection name '{name}' exceeds maximum length of {max_length} characters"
        )
    return name