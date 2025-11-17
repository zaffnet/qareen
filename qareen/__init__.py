"""qareen: A tool for analyzing and optimizing multimodal few-shot example selection."""

import warnings

# Check for GPU availability if torch is installed
try:
    import torch

    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available. For GPU support, please install a CUDA-enabled "
            "PyTorch build from https://pytorch.org/get-started/locally/ before "
            "installing qareen. The package will work with CPU-only PyTorch, but "
            "GPU acceleration will not be available.",
            UserWarning,
            stacklevel=2,
        )
except ImportError:
    # torch not installed yet, which is fine - it will be installed via dependencies
    pass

__version__ = "0.1.0"
