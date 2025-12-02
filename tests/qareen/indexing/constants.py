"""Test constants for indexing tests."""

# Test embedding dimension: 37 chosen as a small prime number unlikely to match real model dims
# (e.g., 512, 768). Using a non-standard dimension helps detect accidental hardcoding of common
# model dimensions in test or implementation code. Small enough for fast tests, large enough to
# catch dimension mismatch bugs. Does not represent actual model dimensions.
TEST_EMBEDDING_DIM: int = 37
