# Setup

Run `uv sync --all-extras` to install dependencies.

## Dependency Management

- `pyproject.toml` is the source of truth. Edit it directly, then run `uv sync`. Never use `uv pip install` or `pip install`.

## Running Python
Use `uv run python ...` for executing python files or commands.

## Architecture Patterns

Pydantic-first: use Pydantic models for all data structures, configs, schemas. Use ABC pattern for extensible components (see `qareen/dataset/base.py`, `qareen/indexing/base.py`). Follow module structure: dataset/, indexing/, config/. Use LangChain VectorStore interface for new vector store backends.

## Code Style

- Type hints required on all functions and methods.
- Docstrings required on all classes and public methods.
- Line length: 100 characters.
- Must pass ruff and mypy.
- Commits: Conventional Commits format.
- **Comments**: Write comments ONLY when absolutely necessary. Your code should be self-documenting. Never comment "WHAT" code does. Only comment "WHY" and that only when diverging from convention or when rationale might be unclear. Avoid verbose comments.
