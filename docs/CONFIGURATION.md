# Configuration

## ChromaDB Telemetry

By default, ChromaDB collects anonymized telemetry data. `qareen` disables this in code, but you can also disable it globally.

### Option 1: Environment Variable (Recommended)

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
export ANONYMIZED_TELEMETRY=False
```

Then reload your shell:

```bash
source ~/.zshrc  # or source ~/.bashrc
```

### Option 2: Per-Command

Prefix any command with the environment variable:

```bash
ANONYMIZED_TELEMETRY=False uv run python scripts/build_index.py ...
```

### Option 3: Programmatic (Already Implemented)

ChromaDB clients are created with telemetry disabled via `qareen.utils.chroma_client.create_chroma_client()`.



## Environment Variables

The `Settings` class (`qareen.models`) reads configuration from environment variables prefixed with `QAREEN_`. You can also place variables in a `.env` file at the project root (uses `pydantic-settings`).

### Reference

| Variable | Type | Req | Default | Description |
|----------|------|:---:|---------|-------------|
| `QAREEN_ENVIRONMENT` | `dev`\|`staging`\|`prod` | ✓ | — | Controls sampling (`dev` uses `dev_sample_size`) |
| `QAREEN_EMBEDDING_MODELS` | JSON array | ✓ | — | Model IDs (CLIP/SIGLIP/Marqo) |
| `QAREEN_ALPHA_VALUES` | JSON array | ✓ | — | Image-text weights [0.0=text, 1.0=image] |
| `QAREEN_DATA_DIR` | path | ✓ | — | Base data directory |
| `QAREEN_CHROMA_DB_DIR` | path | ✓ | — | ChromaDB persistence directory |
| `QAREEN_PREPARED_DATASET_DIR` | path | ✓ | — | Prepared datasets directory |
| `QAREEN_VIZ_OUTPUT_FILE` | path | ✓ | — | Visualization output file |
| `QAREEN_DEV_SAMPLE_SIZE` | int (>0) | ✓ | — | Sample size in dev mode |
| `QAREEN_BATCH_SIZE` | int (>0) | ✓ | — | Indexing batch size |
| `QAREEN_K_NEIGHBORS` | int (>0) | ✓ | — | Retrieval neighbor count |
| `QAREEN_RANDOM_SEED` | int | ✓ | — | Random seed |
| `QAREEN_DATASET_PREP_SAMPLE_SIZE` | int (>0) | ✓ | — | Dataset preparation sample size |
| `QAREEN_REBUILD_COLLECTIONS` | bool | ✓ | — | Delete existing collections before indexing (`scripts/build_index.py`) |
| `QAREEN_DATASET_PATH` | string | — | `None` | Dataset path (local or HuggingFace) |
| `ANONYMIZED_TELEMETRY` | bool | — | `False` | ChromaDB telemetry (no prefix) |

### Usage

```bash
# Minimal dev setup
export QAREEN_ENVIRONMENT="dev" QAREEN_DATA_DIR="data" QAREEN_CHROMA_DB_DIR="chroma_db"
export QAREEN_EMBEDDING_MODELS='["google/siglip-base-patch16-224"]' QAREEN_ALPHA_VALUES='[0.5]'
export QAREEN_DEV_SAMPLE_SIZE="300" QAREEN_BATCH_SIZE="100" QAREEN_K_NEIGHBORS="5"
export QAREEN_RANDOM_SEED="42" QAREEN_DATASET_PREP_SAMPLE_SIZE="1000"
export QAREEN_PREPARED_DATASET_DIR="data/prepared" QAREEN_VIZ_OUTPUT_FILE="data/comparison.md"
export QAREEN_REBUILD_COLLECTIONS="false"
```

## Distance Metric

`qareen` uses **cosine distance** for vector similarity search. This is optimal for comparing normalized embedding vectors from vision-language models.

For detailed information about the distance metric, scoring formula, and rationale, see [DISTANCE_METRIC.md](DISTANCE_METRIC.md).
