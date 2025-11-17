# Dataset Format

qareen expects multimodal datasets that pair a **text** caption with an **image** reference. Each
record MAY optionally include a metadata mapping for auxiliary attributes such as split, task, or
language. Every record SHOULD include a dataset identifier so provenance is preserved throughout the
indexing pipeline.

## Required Fields

| Field | Type | Description |
| --- | --- | --- |
| `text` | `str` | Caption or textual description associated with the sample. |
| `image` | `str \| pathlib.Path \| PIL.Image` | Local path or Pillow image handle for the sample. |

## Optional Fields

| Field | Type | Description |
| --- | --- | --- |
| `metadata` | `dict[str, Any]` | Arbitrary metadata such as split or modality-specific hints. |
| `dataset_name` | `str` | Identifier propagated into collection names. |

## Example Record

```json
{
  "text": "A player shooting a basketball",
  "image": "images/train/0001.png",
  "metadata": {"split": "train"},
  "dataset_name": "sqid"
}
```

The [`DatasetSchema`](../qareen/dataset/schema.py) Pydantic model enforces this shape at load time
and normalizes image paths to strings for serialization friendliness.
