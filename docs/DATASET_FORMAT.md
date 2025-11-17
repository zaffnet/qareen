# Dataset Format

This document describes the expected format for datasets used with qareen.

## Required Fields

All datasets must contain the following fields:

- **`text`**: A non-empty string containing the text content (caption, description, etc.)
- **`image`**: Either a PIL Image object or a path to an image file

## Optional Fields

- **`metadata`**: A dictionary containing additional metadata about the item
- **`dataset_name`**: A string identifier for the dataset

## Example Structure

```python
{
    "text": "A cat sitting on a mat",
    "image": "path/to/image.jpg",  # or PIL.Image.Image object
    "metadata": {
        "split": "train",
        "category": "animals"
    },
    "dataset_name": "my_dataset"
}
```

## HuggingFace Format

When using HuggingFace datasets, the dataset should have `text` and `image` columns. The `image` column can be in HuggingFace's Image feature format.

```python
from datasets import load_dataset

dataset = load_dataset("your_dataset_name")
# Dataset should have 'text' and 'image' columns
```

## Validation Rules

1. **Text**: Must be a non-empty string
2. **Image**: Must be either:
   - A PIL Image object
   - A path string with valid image extension (.jpg, .jpeg, .png, .gif, .bmp, .webp, .tiff, .tif, .svg)
3. **Dataset Name**: Must be sanitizable (lowercase alphanumeric with underscores)

## Collection Naming

Dataset names are sanitized for use in collection names:
- Converted to lowercase
- Special characters replaced with underscores
- Multiple underscores collapsed to single underscore
- Leading/trailing underscores trimmed
- Maximum length: 63 characters (ChromaDB limit)

Example: `"SQID Shots"` → `"sqid_shots"`
