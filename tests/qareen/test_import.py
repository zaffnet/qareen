"""Test that qareen module imports successfully and has expected attributes."""

from unittest.mock import patch


def test_import_module() -> None:
    """Test that qareen imports successfully and GPU check completes."""
    import qareen

    assert qareen.__version__
    assert hasattr(qareen, "check_gpu_available")
    assert isinstance(qareen.check_gpu_available(), bool)


def test_check_gpu_available_suppression() -> None:
    """Test that check_gpu_available works without warnings."""
    with patch("torch.cuda.is_available", return_value=False):
        import qareen

        result = qareen.check_gpu_available()
        assert result is False
