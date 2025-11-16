<!-- 12614abd-33f3-4396-9363-64cce909b572 b3643e85-2911-4c67-8833-29d2e5a35f56 -->
# qareen Project Scaffolding Plan

## Architecture Principles

- Design for 10x scale, redesign at 20x
- Plugin-based architecture for embedding models
- Configuration-driven approach using Pydantic
- Clear separation of concerns (dataset, indexing, retrieval)
- Abstract base classes for extensibility
- Pydantic-first: Use Pydantic models for all data structures, configs, and schemas
- LangChain vector store abstraction for backend flexibility (Chroma → FAISS migration path)

## Directory Structure

```
qareen/
├── qareen/
│   ├── __init__.py
│   ├── dataset/              # Dataset management module
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract base class for dataset loaders
│   │   ├── hf_dataset.py     # HuggingFace dataset implementation
│   │   └── schema.py          # Pydantic models for dataset schema
│   ├── indexing/              # Vector store indexing module
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract base class using LangChain VectorStore
│   │   ├── chroma_indexer.py # ChromaDB implementation via LangChain
│   │   └── models.py          # Embedding model abstractions
│   └── config/                # Configuration management
│       ├── __init__.py
│       └── settings.py        # Pydantic BaseSettings for all config
├── scripts/
│   ├── download_sqid.py      # Script to download SQID dataset
│   └── build_index.py         # Script to build vector store indexes
├── data/                      # Dataset storage (gitignored, cursor ignored)
├── chroma_db/                 # ChromaDB storage (gitignored, cursor ignored)
├── docs/
│   └── DATASET_FORMAT.md     # Documentation on dataset schema/format
├── .gitignore                 # Add data/ and chroma_db/ entries
└── .cursorignore              # Add data/ and chroma_db/ entries
```

## File-by-File Breakdown

### PART 1: Dataset Module

#### `qareen/dataset/__init__.py`

**Purpose:** Public API exports for the dataset module

**Contents:**

- Export `DatasetLoader` abstract base class
- Export `HuggingFaceDatasetLoader` concrete implementation
- Export dataset schema Pydantic models
- Export any dataset-related exceptions

#### `qareen/dataset/base.py`

**Purpose:** Abstract base class defining the dataset loader interface

**Contents:**

- Abstract `DatasetLoader` class (ABC)
- Abstract method `load()`: Returns dataset with validated schema
- Abstract method `validate_schema()`: Validates dataset against expected schema
- Abstract method `get_dataset_name()`: Extracts/returns dataset identifier
- Abstract method `get_dataset_info()`: Returns metadata about the dataset
- All methods use type hints, no implementation

#### `qareen/dataset/hf_dataset.py`

**Purpose:** HuggingFace datasets library implementation

**Contents:**

- `HuggingFaceDatasetLoader` class inheriting from `DatasetLoader`
- Implements all abstract methods from base class
- Uses HuggingFace `datasets` library to load datasets
- Integrates with Pydantic schema validation
- Handles dataset name extraction from HuggingFace dataset info
- Methods: `load()`, `validate_schema()`, `get_dataset_name()`, `get_dataset_info()`
- All methods are stubs with `pass` or minimal structure

#### `qareen/dataset/schema.py`

**Purpose:** Pydantic models defining expected dataset structure

**Contents:**

- Pydantic `BaseModel` classes for dataset schema
- `DatasetSchema`: Main schema model with fields:
  - `text`: Required text field (str)
  - `image`: Required image field (PIL Image or path)
  - `metadata`: Optional dict for additional fields
  - `dataset_name`: Optional str for dataset identifier
- `DatasetItem`: Model for individual dataset items
- Validation rules and field types defined via Pydantic
- Custom validators if needed (all via Pydantic decorators)

### PART 2: Indexing Module

#### `qareen/indexing/__init__.py`

**Purpose:** Public API exports for the indexing module

**Contents:**

- Export `VectorStoreIndexer` abstract base class
- Export `ChromaIndexer` concrete implementation
- Export `EmbeddingModel` abstract base class
- Export any indexing-related exceptions

#### `qareen/indexing/base.py`

**Purpose:** Abstract base class for vector store indexers using LangChain interface

**Contents:**

- Abstract `VectorStoreIndexer` class (ABC)
- Uses LangChain's `VectorStore` as the interface type
- Abstract method `index()`: Takes dataset and creates vector store
- Abstract method `get_collection_name()`: Generates collection name from `dataset_name`, `environment`, `model_id`
  - Pattern: `{environment}_{dataset_name}_{model_id}`
  - Example: `dev_sqid_siglip-base-patch16-224`
- Abstract method `create_vectorstore()`: Creates LangChain VectorStore instance
- Abstract method `get_embeddings()`: Returns LangChain Embeddings instance
- All methods use type hints with LangChain types, no implementation

#### `qareen/indexing/chroma_indexer.py`

**Purpose:** ChromaDB implementation via LangChain

**Contents:**

- `ChromaIndexer` class inheriting from `VectorStoreIndexer`
- Uses `langchain_chroma.Chroma` for vector store creation
- Implements collection naming: `{environment}_{dataset_name}_{model_id}`
- Supports environment parameter (dev/staging/prod)
- Configurable sample size for dev environment (subset of data)
- Methods are stubs with `pass` or minimal structure
- All configuration via Pydantic models

#### `qareen/indexing/models.py`

**Purpose:** Embedding model abstractions

**Contents:**

- Abstract `EmbeddingModel` base class (ABC)
- Integrates with LangChain's `Embeddings` interface
- Abstract method `load_model()`: Loads HuggingFace model
- Abstract method `embed_text()`: Generates text embeddings
- Abstract method `embed_image()`: Generates image embeddings
- Abstract method `embed_multimodal()`: Handles multimodal embedding
- Concrete implementations for HuggingFace models (CLIP, SIGLIP, etc.)
- All models are stubs with type hints, no implementation

### Configuration Module

#### `qareen/config/__init__.py`

**Purpose:** Public API exports for configuration

**Contents:**

- Export `Settings` class (Pydantic BaseSettings)
- Export any configuration-related models

#### `qareen/config/settings.py`

**Purpose:** Centralized configuration using Pydantic BaseSettings

**Contents:**

- `Settings` class inheriting from Pydantic `BaseSettings`
- Fields (all with type hints and defaults):
  - `default_embedding_models`: List[str] - Default model IDs (e.g., ["google/siglip-base-patch16-224"])
  - `data_dir`: Path - Directory for dataset storage (default: "data/")
  - `chroma_db_dir`: Path - Directory for ChromaDB storage (default: "chroma_db/")
  - `dev_sample_size`: int - Number of samples for dev environment (default: 1000)
  - `environment`: Literal["dev", "staging", "prod"] - Current environment
- Environment variable support via Pydantic
- Validation via Pydantic validators
- Settings loading from config files (optional, via Pydantic)

### Scripts

#### `scripts/download_sqid.py`

**Purpose:** CLI script to download SQID dataset

**Contents:**

- Uses `argparse` or `click` for CLI
- Uses HuggingFace `datasets` library
- Downloads SQID dataset
- Saves to `data/` directory (from config)
- Validates dataset schema after download
- Prints dataset info (name, size, etc.)
- Minimal implementation - just structure and argument parsing

#### `scripts/build_index.py`

**Purpose:** CLI script to build vector store indexes

**Contents:**

- Uses `argparse` or `click` for CLI
- Arguments:
  - `--dataset-name`: Required, dataset identifier
  - `--models`: Optional list of model IDs (defaults from config)
  - `--environment`: Optional environment flag (dev/staging/prod, default: dev)
  - `--sample-size`: Optional, override dev sample size
- Loads dataset using `HuggingFaceDatasetLoader`
- For each model:
  - Creates embeddings using model
  - Creates ChromaDB collection via LangChain
  - Collection name: `{environment}_{dataset_name}_{model_id}`
  - Indexes dataset items
- Progress reporting
- Error handling structure
- Minimal implementation - just structure and orchestration logic

### Documentation

#### `docs/DATASET_FORMAT.md`

**Purpose:** Document expected dataset schema and format

**Contents:**

- Required fields: text, image
- Optional fields: metadata
- Dataset name/identifier requirements
- Example dataset structure
- HuggingFace dataset format expectations
- Schema validation rules
- Examples of valid datasets

### Ignore Files

#### `.gitignore` updates

**Purpose:** Exclude data and database directories from git

**Contents:**

- Add `data/` directory
- Add `chroma_db/` directory
- Keep existing ignore patterns

#### `.cursorignore` (new file)

**Purpose:** Exclude data and database from Cursor AI context

**Contents:**

- Add `data/` directory
- Add `chroma_db/` directory
- Pattern matching for data files

## Dependencies

### Update `pyproject.toml`

Add to `[project.dependencies]`:

- `datasets` - HuggingFace datasets library
- `langchain` - Core vector store abstractions
- `langchain-chroma` - ChromaDB integration via LangChain
- `pydantic` - Data validation and settings management
- `pydantic-settings` - Enhanced settings support for Pydantic
- `pillow` - Image handling

**Note:** `langchain-chroma` provides ChromaDB via LangChain interface. For future FAISS support, add `langchain-community` which includes FAISS integration. This keeps architecture backend-agnostic.

## Implementation Approach

### Code Style

- Type hints on all functions and methods
- Docstrings for all classes and public methods
- Abstract base classes with minimal method signatures
- Concrete implementations as stubs (`pass` or minimal structure)
- Pydantic models for all data structures
- Configuration via Pydantic BaseSettings

### No Business Logic

- No actual embedding generation
- No actual indexing implementation
- No dataset processing logic
- Only structure, interfaces, and type definitions
- Focus on extensible architecture

### Collection Naming Convention

- Format: `{environment}_{dataset_name}_{model_id}`
- Examples:
  - `dev_sqid_siglip-base-patch16-224`
  - `staging_sqid_clip-vit-base-patch32`
  - `prod_sqid_siglip-base-patch16-224`
- Sanitization: Replace special characters with underscores
- Validation: Ensure valid collection name format