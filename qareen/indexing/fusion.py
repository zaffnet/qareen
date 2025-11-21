"""Multimodal fusion utilities for combining embeddings and scoring documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


FusionMethod = Literal["linear", "rrf"]
SimilarityMetric = Literal["cosine", "l2", "mmr"]


def _assert_same_dim(image: np.ndarray, text: np.ndarray) -> None:
    if image.shape != text.shape:
        raise ValueError(
            f"Embedding dimension mismatch: image {image.shape}, text {text.shape}"
        )


def linear_combination(
    image: np.ndarray | None, text: np.ndarray | None, alpha: float
) -> np.ndarray:
    """Combine text and image embeddings with linear weighting.

    Args:
        image: Image embedding or None.
        text: Text embedding or None.
        alpha: Image weight (0.0-1.0). Alpha=1.0 drops text contribution.

    Returns:
        L2-normalized combined embedding.

    Raises:
        ValueError: When both embeddings are missing or shapes do not align.
    """

    if image is None and text is None:
        raise ValueError("At least one embedding is required")

    if image is None:
        return _normalize(text)

    if text is None:
        return _normalize(image)

    _assert_same_dim(image, text)
    combined = alpha * image + (1.0 - alpha) * text
    return _normalize(combined)


def _normalize(vector: np.ndarray) -> np.ndarray:
    if vector.ndim == 1:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            raise ValueError("cannot L2-normalize zero vector")
        return vector / norm

    norms = np.linalg.norm(vector, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("cannot L2-normalize zero vector")
    return vector / norms


def cosine_similarity(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """Compute cosine similarity assuming L2-normalized embeddings."""

    return np.dot(documents, query)


def l2_distance(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """Compute L2 distance between query and each document."""

    diffs = documents - query
    return np.linalg.norm(diffs, axis=1)


def reciprocal_rank_fusion(
    image_scores: np.ndarray | None,
    text_scores: np.ndarray | None,
    alpha: float,
    *,
    higher_is_better: bool = True,
    k: int = 60,
) -> np.ndarray:
    """Fuse modality rankings using reciprocal rank fusion.

    Args:
        image_scores: Scores from the image modality or None.
        text_scores: Scores from the text modality or None.
        alpha: Image weight (0.0-1.0). Alpha=1.0 ignores text rankings.
        higher_is_better: Whether larger scores indicate better matches.
        k: Rank offset to dampen tail contributions.

    Returns:
        Fused scores where higher means more relevant.

    Raises:
        ValueError: If both modalities are missing or score lengths mismatch.
    """

    if image_scores is None and text_scores is None:
        raise ValueError("At least one modality score array is required for RRF")

    scores = [s for s in (image_scores, text_scores) if s is not None]
    length = {s.shape[0] for s in scores}
    if len(length) != 1:
        raise ValueError("All score arrays must share the same length")

    weight_image = alpha
    weight_text = 1.0 - alpha

    fused = np.zeros(next(iter(length)))

    def add_modality(values: np.ndarray, weight: float) -> None:
        order = np.argsort(-values) if higher_is_better else np.argsort(values)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, values.shape[0] + 1)
        fused[:] = fused + weight / (k + ranks)

    if image_scores is not None and weight_image:
        add_modality(image_scores, weight_image)
    if text_scores is not None and weight_text:
        add_modality(text_scores, weight_text)

    return fused


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    *,
    lambda_mult: float = 0.5,
    k: int | None = None,
) -> np.ndarray:
    """Select documents using Maximal Marginal Relevance."""

    if k is None:
        k = document_embeddings.shape[0]

    similarity_to_query = np.dot(document_embeddings, query_embedding)
    selected: list[int] = []
    candidates = set(range(document_embeddings.shape[0]))
    while len(selected) < k and candidates:
        if not selected:
            best = int(np.argmax(similarity_to_query[list(candidates)]))
            chosen = list(candidates)[best]
            selected.append(chosen)
            candidates.remove(chosen)
            continue

        candidate_list = list(candidates)
        candidate_similarities = similarity_to_query[candidate_list]
        diversity_penalty = np.max(
            document_embeddings[candidate_list] @ document_embeddings[selected].T,
            axis=1,
        )
        mmr_scores = lambda_mult * candidate_similarities - (1 - lambda_mult) * diversity_penalty
        chosen_idx = int(np.argmax(mmr_scores))
        chosen = candidate_list[chosen_idx]
        selected.append(chosen)
        candidates.remove(chosen)

    return np.array(selected, dtype=int)


@dataclass
class RetrievalConfig:
    alpha: float = 0.5
    fusion: FusionMethod = "linear"
    metric: SimilarityMetric = "cosine"
    mmr_lambda: float = 0.5
    top_k: int | None = None


def rank_documents(
    *,
    query_image: np.ndarray | None,
    query_text: np.ndarray | None,
    doc_images: np.ndarray,
    doc_texts: np.ndarray,
    config: RetrievalConfig,
) -> list[tuple[int, float]]:
    """Rank documents using the configured fusion and similarity strategy."""

    if doc_images.shape != doc_texts.shape:
        raise ValueError("Document embeddings must share the same shape")

    if config.fusion == "linear":
        combined_docs = linear_combination(doc_images, doc_texts, config.alpha)
        combined_query = linear_combination(query_image, query_text, config.alpha)
        if config.metric == "l2":
            distances = l2_distance(combined_query, combined_docs)
            scores = -distances
        elif config.metric == "mmr":
            indices = maximal_marginal_relevance(
                combined_query,
                combined_docs,
                lambda_mult=config.mmr_lambda,
                k=config.top_k,
            )
            scores = np.linspace(1.0, 0.0, num=indices.shape[0], endpoint=True)
            return list(zip(indices.tolist(), scores.tolist(), strict=True))
        else:
            scores = cosine_similarity(combined_query, combined_docs)
    else:
        text_scores = None if query_text is None else cosine_similarity(query_text, doc_texts)
        image_scores = (
            None
            if query_image is None
            else cosine_similarity(query_image, doc_images)
        )
        higher_is_better = True
        if config.metric == "l2":
            text_scores = (
                None
                if query_text is None
                else -l2_distance(query_text, doc_texts)
            )
            image_scores = (
                None
                if query_image is None
                else -l2_distance(query_image, doc_images)
            )

        scores = reciprocal_rank_fusion(
            image_scores,
            text_scores,
            config.alpha,
            higher_is_better=higher_is_better,
        )

    ranking = np.argsort(-scores)
    if config.top_k is not None:
        ranking = ranking[: config.top_k]

    return list(zip(ranking.tolist(), scores[ranking].tolist(), strict=True))
