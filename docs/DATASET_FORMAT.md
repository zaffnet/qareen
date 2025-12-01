# Dataset Format

## Fields

- `text`: Non-empty string or `None`
- `image`: PIL Image, path string, or `None`
- `metadata`: Dict (optional)
- `dataset_name`: String (optional)

At least one of `text` or `image` must be provided.

## Examples

```python
# Dual-modality
{"text": "caption", "image": "path.jpg"}

# Text-only
{"text": "caption", "image": None}

# Image-only
{"text": None, "image": "path.jpg"}

# With metadata
{"text": "caption", "image": "path.jpg", "metadata": {"product_id": "12345", "category": "electronics"}}
```

## Validation

- Both image and text `None`: rejected
- Text must be non-empty when provided
- Image: PIL Image, valid path, or `None`
- Valid extensions: .jpg, .jpeg, .png, .gif, .bmp, .webp, .tiff, .tif, .avif, .heic, .heif, .jfif, .svg

## Embedding

- Dual-modality: alpha-weighted combination using cosine distance
- Single-modality (text or image): alpha ignored
- All embeddings are L2-normalized before storage

For details on similarity scoring, see [DISTANCE_METRIC.md](DISTANCE_METRIC.md).
