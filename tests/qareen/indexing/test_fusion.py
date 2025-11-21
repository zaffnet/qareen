import numpy as np
from PIL import Image

from qareen.indexing.fusion import (
    RetrievalConfig,
    linear_combination,
    maximal_marginal_relevance,
    rank_documents,
    reciprocal_rank_fusion,
)


def test_linear_combination_respects_alpha_extremes() -> None:
    image = np.array([1.0, 0.0])
    text = np.array([0.0, 1.0])

    only_image = linear_combination(image, text, 1.0)
    assert np.allclose(only_image, np.array([1.0, 0.0]))

    only_text = linear_combination(image, text, 0.0)
    assert np.allclose(only_text, np.array([0.0, 1.0]))


def test_reciprocal_rank_fusion_obeys_alpha_weight() -> None:
    text_scores = np.array([0.9, 0.1])
    image_scores = np.array([0.1, 0.9])

    fused_text = reciprocal_rank_fusion(image_scores, text_scores, alpha=0.0)
    fused_image = reciprocal_rank_fusion(image_scores, text_scores, alpha=1.0)

    assert fused_text[0] > fused_text[1]
    assert fused_image[1] > fused_image[0]


def test_rank_documents_cosine_linear_combination() -> None:
    doc_images = np.array([[1.0, 0.0], [0.0, 1.0]])
    doc_texts = np.array([[1.0, 0.0], [0.0, 1.0]])
    query_image = np.array([1.0, 0.0])
    query_text = np.array([1.0, 0.0])

    config = RetrievalConfig(alpha=0.5, fusion="linear", metric="cosine")
    ranking = rank_documents(
        query_image=query_image,
        query_text=query_text,
        doc_images=doc_images,
        doc_texts=doc_texts,
        config=config,
    )

    assert ranking[0][0] == 0


def test_rank_documents_l2_rrf_combination() -> None:
    doc_images = np.array([[0.0, 0.0], [2.0, 0.0]])
    doc_texts = np.array([[1.0, 0.0], [0.0, 1.0]])
    query_image = np.array([0.0, 0.0])
    query_text = np.array([1.0, 0.0])

    config = RetrievalConfig(alpha=0.0, fusion="rrf", metric="l2")
    ranking = rank_documents(
        query_image=query_image,
        query_text=query_text,
        doc_images=doc_images,
        doc_texts=doc_texts,
        config=config,
    )

    assert ranking[0][0] == 0


def test_maximal_marginal_relevance_prefers_diversity() -> None:
    query = np.array([1.0, 0.0])
    docs = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    selected = maximal_marginal_relevance(query, docs, lambda_mult=0.1, k=2)

    assert set(selected.tolist()) == {0, 2}


def test_realistic_small_dataset_pipeline() -> None:
    def text_embedding(text: str) -> np.ndarray:
        tokens = text.lower().split()
        red_score = float(tokens.count("red"))
        blue_score = float(tokens.count("blue"))
        return linear_combination(
            np.array([red_score + 1.0, blue_score + 1.0]),
            np.array([red_score + 1.0, blue_score + 1.0]),
            alpha=0.5,
        )

    def image_embedding(image: Image.Image) -> np.ndarray:
        pixels = np.asarray(image).astype(np.float32) / 255.0
        channel_means = pixels.mean(axis=(0, 1))[:2] + 0.5
        return linear_combination(channel_means, channel_means, alpha=0.5)

    red_image = Image.new("RGB", (4, 4), color=(255, 0, 0))
    blue_image = Image.new("RGB", (4, 4), color=(0, 0, 255))

    doc_images = np.vstack([image_embedding(red_image), image_embedding(blue_image)])
    doc_texts = np.vstack([text_embedding("bright red square"), text_embedding("blue square")])

    query_image = image_embedding(red_image)
    query_text = text_embedding("red square")

    config = RetrievalConfig(alpha=0.6, fusion="rrf", metric="cosine")
    ranking = rank_documents(
        query_image=query_image,
        query_text=query_text,
        doc_images=doc_images,
        doc_texts=doc_texts,
        config=config,
    )

    assert ranking[0][0] == 0
