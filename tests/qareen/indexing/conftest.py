"""Module-level constants for indexing tests."""

# Test embedding dimension constant shared across multiple indexing test modules.
# Usage: Pass this value into model config (e.g., config.projection_dim = TEST_EMBEDDING_DIM)
# and assert that the embedding_dim property equals TEST_EMBEDDING_DIM to verify correct
# property behavior. Value 37 is deliberately non-standard to distinguish test values from
# common production dimensions (e.g., 512, 768) and catch any hardcoded assumptions.
TEST_EMBEDDING_DIM: int = 37
