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

| Variable | Default | Description |
|----------|---------|-------------|
| `ANONYMIZED_TELEMETRY` | `False` | Disable ChromaDB telemetry globally |
| `CHROMA_DB_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `ENVIRONMENT` | `dev` | Environment: `dev`, `staging`, or `prod` |

## Settings

The `Settings` class in `qareen.models` manages configuration:

```python
from qareen.models import Settings

settings = Settings(
    environment="dev",
    chroma_db_dir="./chroma_db",
    dev_sample_size=300,
    batch_size=10,
)
```

### Key Settings

- **environment**: Controls dataset sampling (`dev` uses `dev_sample_size`)
- **dev_sample_size**: Number of samples to use in dev mode (default: 300)
- **batch_size**: Batch size for indexing operations (default: 100)
- **max_image_bytes**: Maximum image size in bytes (default: 10MB)

## Distance Metric

`qareen` uses **cosine distance** for vector similarity search. This is optimal for comparing normalized embedding vectors from vision-language models.

For detailed information about the distance metric, scoring formula, and rationale, see [DISTANCE_METRIC.md](DISTANCE_METRIC.md).
