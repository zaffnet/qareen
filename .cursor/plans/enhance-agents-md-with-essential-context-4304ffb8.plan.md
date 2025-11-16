<!-- 4304ffb8-c536-4e75-ad1b-f104ed25cd16 cb977013-8f43-4473-81de-1eeb5b4f6eb1 -->
# Enhance AGENTS.md for New Feature Development

## Current State

AGENTS.md currently only contains setup instructions. Based on research and user feedback, we need to add minimal but essential context that agents need when implementing NEW features (not documenting existing implementations).

## Key Focus Areas

1. **Dependency Management** - Agents often suggest `uv pip install` or `pip install` instead of editing `pyproject.toml`
2. **Architecture Patterns** - Agents need to know Pydantic-first and ABC patterns when creating new modules
3. **Code Style** - Type hints, docstrings, line length, linting requirements

## Essential Additions

### 1. Dependency Management

- **Source of truth**: `pyproject.toml` is the source of truth for all Python dependencies
- **Adding dependencies**: Edit `pyproject.toml` under `[project.dependencies]` or `[project.optional-dependencies]`, then run `uv sync`
- **Do NOT use**: `uv pip install` or `pip install` - always modify `pyproject.toml` first
- **Python commands**: Use `uv run python ...` or activate venv: `source ~/.zshrc && source .venv/bin/activate && uv run python ...`

### 2. Architecture Patterns (For New Code)

- **Pydantic-first**: When creating new data structures, configs, or schemas, use Pydantic models
- **Abstract base classes**: When creating new extensible components, use ABC pattern (see existing `qareen/dataset/base.py`, `qareen/indexing/base.py` as examples)
- **Module structure**: Follow existing separation of concerns (dataset/, indexing/, config/)
- **LangChain integration**: If adding new vector store backends, use LangChain's VectorStore interface

### 3. Code Style (For New Code)

- **Type hints**: All new functions and methods must have complete type hints
- **Docstrings**: All new classes and public methods require docstrings
- **Line length**: 100 characters maximum (configured in `pyproject.toml`)
- **Linting**: Code must pass ruff and mypy checks (configured in `pyproject.toml`)
- **Commits**: Use Conventional Commits format: `<type>[optional scope]: <description>`
- **Comments**: Write comments ONLY when absolutely necessary. Python is self-documenting and readable. Never comment "WHAT" code does. Only comment "WHY" - when you diverge from convention or when the rationale behind an implementation might be unclear to someone reading the code. Avoid verbose or obvious comments.

## What NOT to Add

- Existing implementation details (GPU checking, collection naming - already implemented)
- Setup instructions for GPU (human responsibility)
- Security/performance/ethical guidelines (too generic, not codebase-specific)
- Testing procedures (covered by existing test structure)

## Critical Constraint

**AGENTS.md is added to LLM context every time** - it must be as short as possible. Every word counts. Be extremely concise and direct.

## Implementation

Add three brief sections to AGENTS.md after Setup:

1. Dependency Management (2-3 lines max)
2. Architecture Patterns (3-4 lines max)  
3. Code Style (4-5 lines max, including strong comment guideline)

Total addition should be ~15-20 lines maximum. Be ruthless about brevity. Focus only on what agents need to know when implementing NEW features.

### To-dos

- [ ] Add Dependency Management section emphasizing pyproject.toml as source of truth and avoiding uv pip install
- [ ] Add Architecture Patterns section for new code (Pydantic-first, ABC pattern, module structure)
- [ ] Add Code Style section for new code (type hints, docstrings, line length, linting)