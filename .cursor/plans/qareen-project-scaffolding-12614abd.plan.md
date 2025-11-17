<!-- 12614abd-33f3-4396-9363-64cce909b572 b3643e85-2911-4c67-8833-29d2e5a35f56 -->
# qareen Project Implementation Plan - Test-Driven Development Edition


## Project Overview

**qareen** (قرين) is Arabic for "constant companion"—a tool for analyzing and optimizing multimodal few-shot example selection for LLMs. It balances relevance, diversity, and modality weighting (image vs. text similarity) when selecting examples.

**TDD Approach**: Review existing tests → Implement code → Verify with test suite → Test end-to-end.

**Deliverables**: Interfaces (ABCs), structure, configuration system, Pydantic schemas, CLI scripts, full implementation.

---

## Key Concepts

### 1. Multimodal Embeddings

Numerical representations capturing semantic meaning. Multimodal models (CLIP, SIGLIP) create embeddings for text and images in the same space, enabling cross-modal similarity.

### 2. Vector Stores

Databases optimized for similarity search (ChromaDB, FAISS). Used for fast nearest neighbor search on large datasets.

### 3. Alpha Parameter

Weight (0.0-1.0) for combining image/text embeddings:

- 0.0 = image-only
- 0.5 = balanced
- 1.0 = text-only

Formula: `V_combined = Normalize(alpha × V_image + (1 - alpha) × V_text)` (both embeddings L2-normalized).

### 4. Collection Naming

Format: `{environment}_{dataset_name}_{model_id}_alpha{alpha_value}`

Example: `dev_sqid_siglip-base-patch16-224_alpha0.50`

Sanitization: lowercase, replace special chars with underscores, collapse multiple underscores, trim.

Alpha formatting: `f"{alpha:.2f}"` (e.g., 0.5 → "0.50").

Validation: Pattern `^[a-z0-9_]+$`, max 63 chars (ChromaDB limit).

### 5. Pre-computation

Pre-compute combined embeddings during indexing (not query-time) for speed and flexibility.

---

## Architecture Principles

1. **Design for 10x scale** (redesign at 20x)
2. **Plugin-based architecture** for embedding models
3. **Configuration-driven** (env vars, config files, defaults)
4. **Separation of concerns** (dataset/indexing/config modules)
5. **Abstract base classes** for extensibility
6. **Pydantic-first** for all data structures
7. **LangChain VectorStore** abstraction (backend-agnostic)
8. **User experience first** (prompt engineer workflows)

---

## Directory Structure

```text
qareen/
├── qareen/
│   ├── __init__.py
│   ├── dataset/
│   │   ├── __init__.py          # Exports: DatasetLoader, HuggingFaceDatasetLoader, schema models
│   │   ├── base.py              # Abstract base class
│   │   ├── hf_dataset.py        # HuggingFace implementation
│   │   └── schema.py            # Pydantic models
│   ├── indexing/
│   │   ├── __init__.py          # Exports: VectorStoreIndexer, ChromaIndexer, EmbeddingModel, exceptions
│   │   ├── base.py              # Abstract base class
│   │   ├── chroma_indexer.py   # ChromaDB implementation
│   │   ├── models.py            # Embedding model abstractions
│   │   └── exceptions.py        # Custom exceptions
│   └── config/
│       ├── __init__.py          # Exports: Settings
│       └── settings.py          # Pydantic BaseSettings
├── scripts/
│   ├── download_sqid.py
│   └── build_index.py
├── data/                         # Gitignored
├── chroma_db/                    # Gitignored
├── docs/
│   └── DATASET_FORMAT.md
├── .gitignore
└── .cursorignore
```

---

## File-by-File Breakdown

### Dataset Module

#### `qareen/dataset/__init__.py`

Exports: `DatasetLoader`, `HuggingFaceDatasetLoader`, `DatasetSchema`, exceptions.

#### `qareen/dataset/base.py`

Abstract `DatasetLoader` class with methods:

- `load()`: Loads dataset, returns standardized format
- `validate_schema()`: Validates required fields (text, image)
- `get_dataset_name()`: Returns dataset identifier
- `get_dataset_info()`: Returns metadata dict/model

#### `qareen/dataset/hf_dataset.py`

Concrete `HuggingFaceDatasetLoader` implementing all abstract methods. Uses HuggingFace `datasets` library, validates with `DatasetSchema`.

#### `qareen/dataset/schema.py`

Pydantic models:

- `DatasetSchema`: Full dataset schema (text, image, optional metadata, optional dataset_name)
- `DatasetItem`: Single item schema
- Validators for image (PIL Image or path) and text (non-empty string)

### Indexing Module

#### `qareen/indexing/__init__.py`

Exports: `VectorStoreIndexer`, `ChromaIndexer`, `EmbeddingModel`, exceptions.

#### `qareen/indexing/exceptions.py`

Custom exceptions:

- `AlphaNotAvailableError`: Query with unavailable alpha (attributes: alpha, available_alphas, model_id, dataset_name, environment)
- `CollectionNameTooLongError`: Name exceeds 63 chars (attributes: collection_name, max_length, suggested_alternatives)
- `InvalidCollectionNameError`: Invalid characters (attributes: collection_name, invalid_characters)

#### `qareen/indexing/base.py`

Abstract `VectorStoreIndexer` with methods:

- `index()`: Creates vector store index (returns LangChain VectorStore)
- `get_collection_name()`: Generates collection name with sanitization/validation
- `create_vectorstore()`: Creates LangChain VectorStore instance
- `get_embeddings()`: Returns LangChain Embeddings instance
- `list_available_alphas()`: Returns sorted list of indexed alpha values
- `validate_alpha_available()`: Checks alpha availability, raises `AlphaNotAvailableError` if not

#### `qareen/indexing/chroma_indexer.py`

Concrete `ChromaIndexer` implementing all abstract methods. Uses `langchain_chroma.Chroma`, supports environments (dev/staging/prod), configurable dev sample size, always rebuilds collections.

#### `qareen/indexing/models.py`

Abstract `EmbeddingModel` base class with methods:

- `load_model()`: Loads HuggingFace model (handles caching, device placement)
- `embed_text(text: str) -> np.ndarray`: Returns L2-normalized text embedding
- `embed_image(image: PIL.Image.Image | str | Path) -> np.ndarray`: Returns L2-normalized image embedding
- `embed_multimodal(image, text, alpha: float) -> np.ndarray`: Combines with alpha weighting (both inputs L2-normalized, alpha in [0.0, 1.0])
- `get_model_id() -> str`: Returns normalized model identifier

Implementations: CLIP, SIGLIP via HuggingFace Transformers.

### Configuration Module

#### `qareen/config/__init__.py`

Exports: `Settings` class.

#### `qareen/config/settings.py`

`Settings` class (Pydantic BaseSettings) with fields:

- `default_embedding_models: List[str]`: Default model IDs
- `default_alpha_values: List[float]`: Default alphas (default: [0.5], validated [0.0, 1.0], deduplicated)
- `data_dir: Path`: Dataset storage (default: "data/", auto-created)
- `chroma_db_dir: Path`: ChromaDB storage (default: "chroma_db/", auto-created)
- `dev_sample_size: int`: Dev samples (default: 1000, positive)
- `environment: Literal["dev", "staging", "prod"]`: Environment (default: "dev", case-insensitive)

Env vars: `QAREEN_{FIELD_NAME}` (list fields: comma-separated).

Precedence: env vars > config file (.env/qareen.env) > defaults.

### Scripts

#### `scripts/download_sqid.py`

CLI script (argparse/click) to download SQID from HuggingFace. Arguments: `--dataset-name`, `--output-dir`, `--validate`, `--sample-size`. Downloads, validates schema, prints info.

#### `scripts/build_index.py`

CLI script to build vector store indexes. Arguments:

- `--dataset-name` (required): Dataset identifier
- `--models` (optional): Model IDs (default: from config, deduplicated)
- `--alpha-values` (optional): Alpha values (default: from config, validated [0.0, 1.0], deduplicated)
- `--environment` (optional): dev/staging/prod (default: dev, case-insensitive)
- `--sample-size` (optional): Override dev sample size
- `--batch-size` (optional): Batch size (default: 100)

Workflow:

1. Load dataset
   1. For each model:

   - Load model once
   - For each alpha: delete existing collection, create new collection
   - Process in batches
   - For each item: compute V_image, V_text, combine with each alpha, store

   1. Report completion

Progress: tqdm with format `[Model: {model_id}] [Alpha: {alpha}] {current}/{total} ({percent}%) ETA: {eta}`.

Error handling: Network retry (3 attempts, exponential backoff), validation errors, file I/O checks, memory suggestions, collection name suggestions.

Logging: Python logging, INFO level, console output, structured format.

### Documentation

#### `docs/DATASET_FORMAT.md`

Required fields: `text` (string), `image` (PIL Image or path). Optional: `metadata` (dict). Dataset name must be sanitizable. Example structure, HuggingFace format expectations, validation rules.

### Ignore Files

#### `.gitignore`

Add: `data/`, `chroma_db/`.

#### `.cursorignore`

Add: `data/`, `chroma_db/`, `*.parquet`, `*.arrow`.

---

## Dependencies

Add to `pyproject.toml` `[project.dependencies]`:

- `datasets`: HuggingFace datasets library
- `langchain`: Vector store abstractions
- `langchain-chroma`: ChromaDB via LangChain
- `pydantic`: Data validation
- `pydantic-settings`: Settings support
- `pillow`: Image handling
- `numpy`: Vector operations
- `tqdm`: Progress bars

---

## Implementation Approach

### TDD Workflow

1. Review existing tests in `tests/`
2. Implement code to pass tests
3. Run `uv run pytest` continuously
4. Iterate until all pass
5. Verify end-to-end with scripts

### Code Style

- Type hints required on all functions/methods
- Docstrings required on all classes/public methods
- ABCs define method signatures with type hints
- Full implementation of all abstract methods
- Pydantic models for all data structures
- Configuration via Pydantic BaseSettings

### Collection Naming

Format: `{environment}_{dataset_name}_{model_id}_alpha{alpha_value}`

Sanitization: lowercase → replace special chars → collapse underscores → trim.

Alpha: `f"{alpha:.2f}"`.

Validation: Pattern `^[a-z0-9_]+$`, max 63 chars.

### Alpha Parameter

**Indexing**: Pre-compute for specified alphas, one collection per (model, alpha). Compute V_image/V_text once, combine with multiple alphas.

**Query**: Validate alpha available (collection exists), raise `AlphaNotAvailableError` if not. No query-time re-indexing.

### Multiple Models/Alphas

Creates collections for all (model, alpha) combinations. Load model once, compute embeddings once per item, combine with each alpha. Process in batches (default: 100).

---

## Testing Strategy

### Philosophy

Use existing tests to guide implementation. Focus on contract compliance, functional correctness, integration, error handling.

### Test Categories

1. **Unit Tests**: Pydantic models, configuration, collection naming, ABCs, exceptions
2. **Integration Tests**: Dataset loading, config integration, CLI parsing, file operations
3. **Contract Tests**: ABC interface compliance, method signatures
4. **CLI Tests**: Argument parsing, validation, error messages
5. **Error Handling Tests**: Custom exceptions, error context, user guidance

### Test Infrastructure

- Framework: pytest
- Mocking: External services, file system, environment, vector stores
- Coverage: ≥90% for implementation code
- CI: Automated on every commit

### Testing Workflow

1. Review tests in `tests/qareen/` and `tests/scripts/`
2. Run `uv run pytest` to see failures
3. Implement code to pass tests
4. Run tests frequently during development
5. Refactor while maintaining coverage
6. Verify end-to-end: `uv run python scripts/download_sqid.py --sample-size 100` then `uv run python scripts/build_index.py --dataset-name sqid --sample-size 100`

---

## Additional Considerations

### Logging

Python logging, INFO level, console output, format: `{timestamp} [{level}] {module}: {message}`. Log: model loading, collection operations, indexing progress, errors, config loading.

### Error Handling

- Network: Retry with exponential backoff (3 attempts)
- Validation: Clear messages with suggestions
- File I/O: Check permissions, disk space, paths
- Memory: Suggest reducing batch/sample size
- Collection names: Suggest shorter names
- Alpha: Normalize to 2 decimal places before comparison

### Performance

- Batch processing (configurable, default: 100)
- Model caching (reuse across alphas)
- Progress reporting (tqdm)
- Memory-efficient embedding computation

---

## Verification Steps

### Step 1: Download Dataset

```bash
uv run python scripts/download_sqid.py --dataset-name sqid --sample-size 100
```

Verify: Dataset in `data/`, has text/image fields, ~100 samples.

### Step 2: Build Index

```bash
uv run python scripts/build_index.py --dataset-name sqid --sample-size 100
```

Verify: Collection in `chroma_db/`, name follows convention `dev_sqid_{model_id}_alpha0.50`, contains embeddings.

### Step 3: Multiple Alphas

```bash
uv run python scripts/build_index.py --dataset-name sqid --alpha-values 0.0 0.5 1.0 --sample-size 100
```

Verify: Three collections exist with correct names (`_alpha0.00`, `_alpha0.50`, `_alpha1.00`), same item count.

### Step 4: Full Test Suite

```bash
uv run pytest
```

Verify: All tests pass, no linting/type errors.

---

## Summary

TDD implementation: Review tests → Implement code → Verify with test suite → Test end-to-end.

Creates: Interfaces (ABCs), structure, configuration, Pydantic schemas, CLI scripts, full implementation.

Verification: Download dataset (100 samples) and build index to verify end-to-end functionality.