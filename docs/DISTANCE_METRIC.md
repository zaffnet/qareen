# Distance Metric

`qareen` uses **cosine distance** for ChromaDB vector search and exposes additional reranking
strategies for multimodal retrieval.

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

## Additional retrieval options

- **Reciprocal Rank Fusion (RRF):** Blend independent text and image rankings with an `alpha`
  weight (alpha=1.0 ignores text). Useful when modalities disagree but both signals matter.
- **Maximum Marginal Relevance (MMR):** Configurable `lambda` balances similarity and diversity
  on combined embeddings.
- **L2 distance:** Optional metric for experiments that prefer Euclidean spacing over angular
  similarity.

## L2 Normalization Still Required

**Yes**, L2 normalization is still needed:
- Embeddings must be normalized before storage: `embedding / ||embedding||`
- Cosine distance assumes normalized vectors
- All models apply normalization in their `embed_*` methods
