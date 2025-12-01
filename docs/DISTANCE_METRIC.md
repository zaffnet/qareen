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

qareen models automatically L2-normalize embeddings before storage, so developers do not need to normalize manually.

> [!NOTE]
> Manual normalization (`embedding / ||embedding||`) is only necessary if embeddings are produced or stored outside of qareen's APIs.
