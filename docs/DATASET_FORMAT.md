# Dataset Format

## Required Fields

At least one of:
- `text`: Non-empty string
- `image`: PIL Image or path string

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
```

## Validation

- Both `None` rejected
- Text must be non-empty when provided
- Image: PIL Image, valid path, or `None`
- Valid extensions: .jpg, .jpeg, .png, .gif, .bmp, .webp, .tiff, .tif, .svg

## Embedding

- Dual-modality: alpha-weighted combination
- Single-modality: alpha ignored
