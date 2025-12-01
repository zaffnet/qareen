from __future__ import annotations

import hashlib
import re

# ChromaDB collection name max length limit
CHROMA_COLLECTION_MAX_LENGTH = 63

# Regex pattern for parsing alpha values from collection names (e.g., "_a0_500" -> "0_500")
ALPHA_SUFFIX_PATTERN = re.compile(r"_a(\d+_\d+)")


def get_collection_name(
    dataset_name: str, model_id: str, alpha: float | None = None, environment: str = "dev"
) -> str:
    dataset_name = dataset_name.strip()
    model_id = model_id.strip()
    env = environment.strip().lower()

    if not dataset_name:
        raise ValueError("Input 'dataset_name' cannot be empty.")
    if not model_id:
        raise ValueError("Input 'model_id' cannot be empty.")
    if env not in ("dev", "staging", "prod"):
        raise ValueError(
            f"Invalid environment: '{environment}'. Must be 'dev', 'staging', or 'prod'."
        )

    def sanitize(part: str) -> str:
        return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", part.lower())).strip("_")

    env_part, dataset_part, model_part = sanitize(env), sanitize(dataset_name), sanitize(model_id)
    if not dataset_part:
        raise ValueError("invalid sanitized dataset_name")
    if not model_part:
        raise ValueError("invalid sanitized model_id")

    if alpha is not None:
        alpha_str = f"{alpha:.3f}".replace(".", "_")
        alpha_suffix = f"_a{alpha_str}"
    else:
        alpha_suffix = ""
    max_length = CHROMA_COLLECTION_MAX_LENGTH
    available_length = max_length - len(env_part) - len(alpha_suffix) - 2

    if len(dataset_part) + len(model_part) <= available_length:
        base_name = f"{env_part}_{dataset_part}_{model_part}"
    else:
        hash_hex = hashlib.sha256(f"{dataset_part}_{model_part}".encode()).hexdigest()[:8]
        hash_suffix = f"_h{hash_hex}"
        available_for_parts = available_length - len(hash_suffix)
        if available_for_parts <= 0:
            raise ValueError("length budget exceeded")
        half_available = available_for_parts // 2
        dataset_truncated = dataset_part[:half_available]
        model_truncated = model_part[: available_for_parts - len(dataset_truncated)]
        base_name = f"{env_part}_{dataset_truncated}_{model_truncated}{hash_suffix}"

    name = f"{base_name}{alpha_suffix}"
    if len(name) > max_length:
        raise ValueError("collection name too long")
    return name
