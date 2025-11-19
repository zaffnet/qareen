"""Test that qareen module imports successfully and has expected attributes."""

import inspect
import warnings
from unittest.mock import patch


def test_import_module() -> None:
    """Test that qareen imports successfully and GPU check completes."""
    import qareen

    assert qareen.__version__
    assert hasattr(qareen, "check_gpu_available")
    assert isinstance(qareen.check_gpu_available(), bool)


def _call_check_gpu_available() -> None:
    """Helper function to call check_gpu_available from this test file."""
    import qareen

    qareen.check_gpu_available()


def test_gpu_warning_stacklevel() -> None:
    """Test that GPU warning points to the caller, not the function itself."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        warnings.catch_warnings(record=True) as w,
    ):
        warnings.simplefilter("always")
        frame = inspect.currentframe()
        if frame is None:
            try:
                stack = inspect.stack()
                frame = stack[1].frame if len(stack) > 1 else None
            except (IndexError, AttributeError):
                frame = None
        call_line = frame.f_lineno + 1 if frame else 0
        _call_check_gpu_available()

        assert len(w) > 0, "Warning should be emitted when CUDA is not available"
        warning = w[0]
        assert warning.category is UserWarning
        assert "CUDA is not available" in str(warning.message)
        assert warning.filename == __file__
        assert warning.lineno == call_line
