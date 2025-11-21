# Distance Metrics

`qareen` supports **cosine distance** (default) and **L2 distance** for ChromaDB vector search.

## Cosine Similarity (Default)

Formula:
```python
similarity_score = max(0.0, min(1.0, 1.0 - (cosine_distance / 2.0)))
```

Score range: [0.0, 1.0] where 1.0 = identical, 0.5 = orthogonal, 0.0 = opposite.

## Why Cosine?

All embedding models produce L2-normalized vectors. Cosine distance:
- Measures directional similarity (angle between vectors)
- Provides meaningful scores across all alpha values (0.0 to 1.0)
- Aligns with vision-language model training objectives

Previous L2 distance with quadratic penalty caused zero scores for image-only queries.

## L2 Distance

Optionally, you can use L2 (Euclidean) distance. This metric is strictly distance-based (lower is better).

```python
score = l2_distance
```

## Reciprocal Rank Fusion (RRF)

For combining rankings from different modalities (text vs image) without linear embedding combination, `qareen` supports Weighted RRF:

```python
score = alpha * (1 / (k + rank_image)) + (1 - alpha) * (1 / (k + rank_text))
```

## L2 Normalization Still Required

**Yes**, L2 normalization is still needed:
- Embeddings must be normalized before storage: `embedding / ||embedding||`
- Cosine distance assumes normalized vectors
- All models apply normalization in their `embed_*` methods
