<!-- 14be16eb-5112-43c2-a01e-f9016b2f7e88 e9d71e0e-af65-4402-92c9-f5ff097bb376 -->
# Fix Critical Multimodal Query Bug

## Problem Summary

**Critical architectural flaw**: Indexing correctly stores multimodal embeddings (`alpha * image_emb + (1-alpha) * text_emb`), but retrieval uses text-only queries. This completely negates the purpose of alpha-weighted multimodal indexing.

**Root cause**: `EmbeddingModelWrapper.embed_query()` only calls `embed_text()`, so LangChain's `similarity_search()` methods cannot perform multimodal queries.

## Implementation Plan

### 1. Add Multimodal Query Support to ChromaIndexer

**File**: `qareen/indexing/chroma_indexer.py`

Add new method to ChromaIndexer:

```python
def query_multimodal(
    self,
    vectorstore: VectorStore,
    image: Image.Image | str | None,
    text: str | None,
    alpha: float,
    k: int = 5,
    score_threshold: float | None = None,
) -> list[tuple[Document, float]]:
    """Query vectorstore with multimodal embedding.

    Args:
        vectorstore: VectorStore instance to query
        image: Query image (PIL Image or URL)
        text: Query text
        alpha: Alpha value matching the indexed collection
        k: Number of results to return
        score_threshold: Optional minimum similarity score

    Returns:
        List of (Document, score) tuples
    """
```

**Implementation**:

- Load and validate image using `_load_image()`
- Call `self.embedding_model.embed_multimodal(image, text, alpha)`
- Use Chroma's underlying collection to search by vector
- Return results with scores

**Key considerations**:

- Works with all embedding models (SigLIP, Marqo, future models)
- Validates alpha matches collection alpha (metadata check)
- Handles text-only, image-only, or multimodal queries
- Properly handles image loading/downloading

### 2. Fix visualize_marqo_comparison.py

**File**: `scripts/visualize_marqo_comparison.py`

**Current bug** (line 201):

```python
results = vectorstore.similarity_search_with_score(query_text, k=k)
```

**Fix**:

```python
results = indexer.query_multimodal(
    vectorstore=vectorstore,
    image=query_image,
    text=query_text,
    alpha=alpha,
    k=k,
)
```

### 3. Comprehensive Testing

#### A. Unit Tests (`tests/qareen/indexing/test_multimodal_query.py` - NEW FILE)

Test `ChromaIndexer.query_multimodal()`:

- Text-only query with text-only index
- Image-only query with image-only index
- Multimodal query with multimodal index
- Different alpha values produce different results with different scores
- Alpha 0.0 (text-only) vs alpha 1.0 (image-only) return different results
- Validates image loading (PIL, URL, local path)
- Validates query with both SigLIP and Marqo models
- Error handling for invalid inputs

**Critical test**: Verify that queries with different alphas return different samples with different scores on the same dataset.

#### B. Integration Tests (`tests/qareen/indexing/test_alpha_query_integration.py` - NEW FILE)

Compare retrieval results across alpha spectrum:

- Index small dataset with alphas [0.0, 0.5, 1.0]
- Query with same multimodal input across all alphas
- Assert results differ significantly between alpha=0.0 and alpha=1.0
- Assert alpha=0.5 results are distinct from both extremes
- Verify score distributions differ
- Test with real image+text samples

#### C. Regression Tests (`tests/qareen/indexing/test_query_regression.py` - NEW FILE)

Fixed expected results to prevent silent breakage:

- Create small fixed dataset (3-5 samples with known images/text)
- Index with known alpha values
- Query with specific image+text
- Assert top-k results match expected indices
- Assert scores are within expected ranges
- Freeze with test fixtures to detect any future changes

#### D. Update Existing Tests

**File**: `tests/scripts/test_visualize_marqo_comparison.py`

Update mocks to ensure `query_multimodal()` is called, not `similarity_search()`.

### 4. Documentation Updates

**File**: `qareen/indexing/chroma_indexer.py`

Add comprehensive docstring to `query_multimodal()` with usage examples.

**File**: `scripts/visualize_marqo_comparison.py`

Add comment explaining why multimodal query is necessary.

### 5. Self-Verification Checklist

Before completion:

1. Run all new tests - confirm they fail on old code, pass on new code
2. Run full test suite - ensure no regressions
3. Run pre-commit hooks - ensure code quality
4. Manually test visualize_marqo_comparison.py with different alphas
5. Verify results differ between alpha=0.0 and alpha=1.0
6. Check that both SigLIP and Marqo models work correctly
7. Confirm no other scripts have the same bug (search for `similarity_search` usage)

## Files to Modify

1. `qareen/indexing/chroma_indexer.py` - Add `query_multimodal()` method
2. `scripts/visualize_marqo_comparison.py` - Fix query logic
3. NEW: `tests/qareen/indexing/test_multimodal_query.py`
4. NEW: `tests/qareen/indexing/test_alpha_query_integration.py`
5. NEW: `tests/qareen/indexing/test_query_regression.py`
6. `tests/scripts/test_visualize_marqo_comparison.py` - Update mocks

## Expected Outcome

After fix:

- Queries use both image and text (weighted by alpha)
- Different alpha values return different results with different scores
- Multimodal retrieval actually works as designed
- Comprehensive tests prevent future regressions
- All experiments and visualizations will be valid
