# Distance Metric

`qareen` uses **cosine distance** for ChromaDB vector search.

## Formula

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

## L2 Normalization Still Required

**Yes**, L2 normalization is still needed:
- Embeddings must be normalized before storage: `embedding / ||embedding||`
- Cosine distance assumes normalized vectors
- All models apply normalization in their `embed_*` methods
