# Test Review: Single-Modality Support

## Overview
Tests define behavior for dataset samples with only text OR only image (not both required).

## Test Files

### 1. `tests/qareen/dataset/test_schema.py`
- `test_dataset_schema_accepts_text_only`: Accept text-only samples (image=None)
- `test_dataset_schema_accepts_image_only`: Accept image-only samples (text=None)
- `test_dataset_schema_accepts_pil_image_only`: Accept PIL Image with text=None
- `test_dataset_schema_rejects_both_none`: Reject when both are None
- `test_dataset_schema_rejects_empty_text_when_image_none`: Reject empty text as only modality

### 2. `tests/qareen/indexing/test_single_modality.py`
- `test_embedding_model_returns_none_for_missing_text`: embed_text(None) returns None
- `test_embedding_model_returns_none_for_missing_image`: embed_image(None) returns None
- `test_embedding_model_returns_text_only_embedding`: embed_multimodal with image=None works
- `test_embedding_model_returns_image_only_embedding`: embed_multimodal with text=None works
- `test_embedding_model_rejects_both_none`: embed_multimodal rejects both None
- `test_indexer_handles_text_only_samples`: Index and retrieve text-only samples
- `test_indexer_handles_image_only_samples`: Index and retrieve image-only samples
- `test_indexer_handles_mixed_modality_samples`: Index mixed modality dataset

### 3. `tests/qareen/indexing/test_alpha_single_modality.py`
- `test_alpha_ignored_for_text_only_samples`: Alpha passed but ignored for text-only
- `test_alpha_ignored_for_image_only_samples`: Alpha passed but ignored for image-only
- `test_alpha_matters_for_dual_modality_samples`: Alpha used when both present
- `test_multiple_alphas_with_single_modality`: Multiple alpha values work with single modality
- `test_text_only_query_with_text_only_index`: Text query retrieves text-only samples
- `test_image_query_embedding_with_image_only_index`: Query works on image-only index
- `test_alpha_zero_equivalent_to_text_only`: Alpha=0.0 uses text only
- `test_alpha_one_equivalent_to_image_only`: Alpha=1.0 uses image only

## Expected Behavior

### Schema Layer
- DatasetItem must accept `text=None` OR `image=None` (at least one required)
- Must reject when both are None with "at least one modality" error message
- Empty/whitespace text rejected when image=None

### Embedding Layer
- embed_text(None) and embed_image(None) return None
- embed_multimodal handles None gracefully:
  - If image=None: return text embedding only
  - If text=None: return image embedding only
  - If both None: raise ValueError
- Alpha parameter passed through but ignored for single-modality cases

### Indexing Layer
- ChromaIndexer must index samples with missing modalities
- Store whichever embedding exists (text or image)
- Retrieval works correctly with single-modality indices
- Multiple alpha values work even with single-modality samples

## Test Status
All tests currently FAIL (expected). Implementation required to pass tests.

## Review Approval
Once approved, tests become immutable. Implementation phase cannot modify tests.
