# Dataset Format

This document outlines the expected schema and format for datasets used with the `qareen` library.

## Schema

All datasets must adhere to a specific schema, which is enforced by the `qareen.dataset.schema.DatasetSchema` Pydantic model. The schema consists of a list of `DatasetItem` objects, each with the following fields:

- `text` (str, required): The text content associated with the data point.
- `image` (str or PIL.Image, required): The image content. This can be a path to an image file or a PIL Image object.
- `metadata` (dict, optional): A dictionary of additional metadata for the data point.

### Dataset Name

The `DatasetSchema` also has an optional `dataset_name` field, which is a string identifier for the dataset.

## Example

Here is an example of a valid dataset structure:

```json
{
  "dataset_name": "my_awesome_dataset",
  "data": [
    {
      "text": "This is the first item.",
      "image": "path/to/image1.png",
      "metadata": {
        "category": "A"
      }
    },
    {
      "text": "This is the second item.",
      "image": "path/to/image2.png"
    }
  ]
}
```

## HuggingFace Datasets

When using datasets from the HuggingFace Hub, the dataset must contain `text` and `image` columns. Any other columns will be ignored.
