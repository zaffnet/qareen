"""Shared test fixtures and constants for indexing tests."""

# Arbitrary embedding dimension for testing embedding_dim property.
# The actual value doesn't matter - we're testing that the property
# correctly reads from model config. Using a non-standard dimension
# (not 512 or 768) to clearly show it's a test value.
TEST_EMBEDDING_DIM = 63
