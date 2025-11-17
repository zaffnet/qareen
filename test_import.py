"""Test that qareen module imports successfully and has expected attributes."""

import warnings


def test_import_module() -> None:
    """Test that qareen imports successfully and GPU check completes."""
    import qareen

    assert qareen.__version__
    assert hasattr(qareen, "check_gpu_available")
    assert isinstance(qareen.check_gpu_available(), bool)


def test_gpu_warning_stacklevel() -> None:
    """Test that GPU warning points to the caller, not the function itself."""
    import qareen

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        qareen.check_gpu_available()

        if len(w) > 0:
            warning = w[0]
            assert warning.category is UserWarning
            assert "CUDA is not available" in str(warning.message)
            assert warning.filename == __file__
            assert warning.lineno > 0
