# Local & Custom Data: Storage, Indexing & Retrieval

## Storing Local Data

Save your dataset using HuggingFace's dataset format:

```python
from datasets import Dataset
from PIL import Image

# Prepare your data (must have 'text' and 'image' fields)
data = [
    {"text": "red leather handbag", "image": Image.open("bag1.jpg")},
    {"text": "blue cotton dress", "image": Image.open("dress1.jpg")},
    {"text": "black running shoes", "image": Image.open("shoes1.jpg")},
]

# Create and save dataset
dataset = Dataset.from_list(data)
dataset.save_to_disk("data/my_products")
```

**Required fields:**
- `text`: string or `None`
- `image`: PIL Image, path string, or `None`

At least one modality must be present. See [DATASET_FORMAT.md](DATASET_FORMAT.md).

## Indexing Data

Index your local dataset using the `build_index.py` script:

```bash
uv run python scripts/build_index.py \
  --dataset-name data/my_products \
  --models google/siglip-base-patch16-224 \
  --alpha-values 0.0 0.5 1.0 \
  --environment dev \
  --batch-size 100
```

**Key parameters:**
- `--dataset-name`: Path to local dataset directory (or HuggingFace Hub ID)
- `--models`: One or more embedding models (CLIP, SIGLIP, Marqo variants)
- `--alpha-values`: Image-text weights (0.0=text-only, 1.0=image-only)
- `--environment`: dev/staging/prod (affects collection naming)
- `--rebuild`: Add to delete and recreate existing collections

The script will:
1. Load your dataset from disk
2. Validate schema (text + image fields)
3. Generate embeddings for each alpha value
4. Store in ChromaDB vector store

**Example with multiple models:**

```bash
uv run python scripts/build_index.py \
  --dataset-name data/my_products \
  --models openai/clip-vit-large-patch14 Marqo/marqo-fashionSigLIP \
  --alpha-values 0.0 0.25 0.5 0.75 1.0 \
  --environment prod
```

## Retrieval / Querying

Use the `ChromaRetriever` to query indexed data:

```python
from PIL import Image
from qareen.models import Settings
from qareen.indexing.siglip_model import SIGLIPEmbeddingModel
from qareen.retrieving.chroma_retriever import ChromaRetriever

settings = Settings(environment="dev")
embedding_model = SIGLIPEmbeddingModel("google/siglip-base-patch16-224")
retriever = ChromaRetriever(embedding_model=embedding_model, settings=settings)

vectorstore = retriever.get_vectorstore(
    dataset_name="my_products",
    model_id="google/siglip-base-patch16-224",
    alpha=0.5,
    environment="dev",
)

query_image = Image.open("query_image.jpg")
results = retriever.query_multimodal(
    vectorstore=vectorstore,
    image=query_image,
    text="designer handbag",
    alpha=0.5,
    k=5,
)

for doc, score in results:
    print(f"Score: {score:.3f} | Text: {doc.page_content} | Metadata: {doc.metadata}")
```

**Query variations:**

```python
# Text-only query (alpha=0.0)
results = retriever.query_multimodal(vectorstore=vectorstore_alpha_0, image=None, text="leather handbag", alpha=0.0, k=5)

# Image-only query (alpha=1.0)
results = retriever.query_multimodal(vectorstore=vectorstore_alpha_1, image=query_image, text=None, alpha=1.0, k=5)

# Balanced multimodal (alpha=0.5)
results = retriever.query_multimodal(vectorstore=vectorstore_alpha_05, image=query_image, text="red handbag", alpha=0.5, k=10)
```

## Quick Start

1. **Prepare**: Create dataset with `text`/`image` columns and save to disk.
2. **Index**: Run `uv run python scripts/build_index.py --dataset-name <path> ...`
3. **Query**: Use `ChromaRetriever.query_multimodal()` in your application.

## Configuration

Environment variables and paths are managed via `Settings`:

```python
from qareen.models import Settings

settings = Settings(
    environment="prod",
    chroma_db_dir=Path("/custom/path/chroma_db"),
    dev_sample_size=1000,
)
```

See [CONFIGURATION.md](CONFIGURATION.md) for full configuration options.

## Notes

- **Alpha values:** Must match between indexing and querying (use same alpha)
- **Collection naming:** Format: `{env}_{dataset}_{model}_a{alpha:.3f}`
- **Storage:** ChromaDB persists to `chroma_db/` directory by default
- **Models:** Any HuggingFace CLIP/SIGLIP model supported
- **Distance metric:** Cosine distance (see [DISTANCE_METRIC.md](DISTANCE_METRIC.md))
