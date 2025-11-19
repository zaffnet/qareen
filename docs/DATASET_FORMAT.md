# Dataset Format

## Required Fields

At least one of `text` or `image` must be provided. Each field may be `None` individually, but not both simultaneously.

- `text`: Non-empty string or `None`
- `image`: PIL Image, path string, or `None`

## Optional Fields

- `metadata`: Dict
- `dataset_name`: String

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
- Valid extensions: .jpg, .jpeg, .png, .gif, .bmp, .webp, .tiff, .tif, .svg

## Embedding

- Dual-modality: alpha-weighted combination
- Single-modality (text or image): alpha ignored
