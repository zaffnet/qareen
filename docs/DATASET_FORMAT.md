# Dataset Format

The dataset should be a HuggingFace dataset with the following fields:

- `text`: A string containing the text of the item.
- `image`: A PIL Image object or a path to an image file.
- `metadata`: An optional dictionary containing any additional metadata.

The dataset name should be sanitizable, meaning it should not contain any special characters that would be problematic for file or collection names.
