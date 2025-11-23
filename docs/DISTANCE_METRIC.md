# Distance Metric

`qareen` uses **cosine distance** for ChromaDB vector search.

## Formula

```python
similarity_score = max(0.0, min(1.0, 1.0 - (cosine_distance / 2.0)))
```

Score range: [0.0, 1.0] where 1.0 = identical, 0.5 = orthogonal, 0.0 = opposite.

## Why Cosine?

Cosine distance measures directional similarity, providing consistent scores [0.0, 1.0] across all alpha values for L2-normalized vectors. This aligns with vision-language model training objectives.

## L2 Normalization

Embeddings must be L2-normalized before storage (`embedding / ||embedding||`) as cosine distance assumes normalized vectors. All `qareen` models handle this automatically.
