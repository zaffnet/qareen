# Setup

Run `uv sync --all-extras` to install dependencies.

## Dependency Management

- `pyproject.toml` is the source of truth. Edit it directly, then run `uv sync`. Never use `uv pip install` or `pip install`.

## Running Python
Use `uv run python ...` for executing python files or commands.

## Architecture Patterns

Pydantic-first: use Pydantic models for all data structures, configs, schemas. Use ABC pattern for extensible components (see `qareen/dataset/base.py`, `qareen/indexing/base.py`). Follow module structure: dataset/, indexing/, config/. Use LangChain VectorStore interface for new vector store backends.

## Code Style

- Type hints required on all functions and methods you modify or add.
- Docstrings required on all classes and public methods.
- Line length: 100 characters.
- Must pass ruff and mypy.
- Commits: Conventional Commits format.
- **Comments**: Write comments ONLY when absolutely necessary. Your code should be self-documenting. Never comment "WHAT" code does. Only comment "WHY" and that only when diverging from convention or when rationale might be unclear. Avoid verbose comments.

## Communication & Code: Minimalism Required

**Rules:**
1. **Shut up and code** - No preambles, explanations, or narration
2. **Edit, don't create** - Always prefer editing existing files over creating new ones
3. **One task = one change** - Don't add features that weren't requested
4. **No obvious comments** - Code should be self-documenting
5. **No helper files** - No utils.py, helpers.py, or "temporary" scripts unless explicitly asked

**Anti-patterns:**
- ❌ Creating new files when editing would work
- ❌ Adding logging/error handling that wasn't requested
- ❌ Creating documentation files (README, CONTRIBUTING, etc.) proactively
- ❌ Writing test helpers or fixtures that weren't needed
- ❌ Refactoring code that works and wasn't mentioned
- ❌ Adding type hints to files you didn't touch
- ❌ "Improving" variable names in passing

**Speaking rules:**
- Errors: state problem + fix concisely; include enough context to reproduce (logs, steps, inputs) when relevant
- Clarifications: ask question directly; brief context or examples are permitted to aid understanding
- Completion: "✅ [what was done]" with 1–3 sentences; optional note about caveats or next steps for handoff

## ⚠️ CRITICAL: Pre-commit Checks - REQUIRED BEFORE COMPLETION

**Before declaring work complete, you MUST:**

1. Run: `uv run pre-commit run --all-files`
2. If ANY hooks fail: fix issues and re-run step 1
3. Repeat until ALL hooks show "Passed"
4. Explicitly state: "✅ Ran `uv run pre-commit run --all-files` - ALL HOOKS PASSED"

**Note:** Never assume checks pass. Always run and confirm. CI/CD will reject PRs if this fails.
