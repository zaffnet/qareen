"""Naming utilities for collections and resources."""

from __future__ import annotations

import hashlib
import re


def get_collection_name(
    dataset_name: str,
    model_id: str,
    alpha: float | None = None,
    environment: str = "dev",
) -> str:
    """Generate sanitized collection name.

    Format: {environment}_{dataset_name}_{model_id}_alpha{alpha_value}

    Args:
        dataset_name: Dataset identifier
        model_id: Model identifier
        alpha: Alpha value (optional, formatted to 3 decimals if provided)
        environment: Environment (dev/staging/prod)

    Returns:
        Sanitized collection name

    Raises:
        ValueError: If inputs are invalid or name exceeds 63 characters
    """
    dataset_name = dataset_name.strip()
    if not dataset_name:
        raise ValueError("dataset_name must be a non-empty string")

    model_id = model_id.strip()
    if not model_id:
        raise ValueError("model_id must be a non-empty string")

    environment = environment.strip()
    env = environment.lower()
    if env not in ("dev", "staging", "prod"):
        raise ValueError(
            f"environment must be one of 'dev', 'staging', or 'prod', got '{environment}'"
        )

    sanitized_parts = []
    for part in [env, dataset_name, model_id]:
        sanitized = part.lower()
        sanitized = re.sub(r"[^a-z0-9_]+", "_", sanitized)
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = sanitized.strip("_")
        sanitized_parts.append(sanitized)

    env_part, dataset_part, model_part = sanitized_parts

    alpha_suffix = f"_a{alpha:.3f}" if alpha is not None else ""
    max_length = 63

    available_length = max_length - len(env_part) - len(alpha_suffix) - 2

    if len(dataset_part) + len(model_part) <= available_length:
        base_name = f"{env_part}_{dataset_part}_{model_part}"
    else:
        hash_suffix_len = 10
        hash_input = f"{dataset_part}_{model_part}"
        hash_hex = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        hash_suffix = f"_h{hash_hex}"

        available_for_parts = available_length - hash_suffix_len
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
