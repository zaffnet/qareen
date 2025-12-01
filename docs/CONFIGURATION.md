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

The `ChromaIndexer` class already disables telemetry when creating ChromaDB clients:

```python
from chromadb.config import Settings as ChromaSettings

chroma_client = chromadb.PersistentClient(
    path=str(settings.chroma_db_dir),
    settings=ChromaSettings(anonymized_telemetry=False),
)
```

## Logging Configuration

### Basic Setup

By default, `qareen` uses rich-formatted logging. To configure logging behavior:

```python
from qareen.indexing.chroma_indexer import setup_logging

# Rich formatting (default)
setup_logging(rich=True, level=logging.INFO)

# Plain text output
setup_logging(rich=False, level=logging.INFO)

# Debug level
setup_logging(rich=True, level=logging.DEBUG)
```

### Disabling Rich Output

For CI/CD pipelines or environments where rich formatting causes issues, use plain logging:

```python
setup_logging(rich=False)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANONYMIZED_TELEMETRY` | `False` | Disable ChromaDB telemetry globally |
| `CHROMA_DB_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `ENVIRONMENT` | `dev` | Environment: `dev`, `staging`, or `prod` |

## Settings

The `Settings` class in `qareen.config.settings` manages configuration:

```python
from qareen.config.settings import Settings

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
