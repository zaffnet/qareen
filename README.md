# qareen

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-F7931E?logo=gradio&logoColor=white)](https://www.gradio.app/)
[![HF Transformers](https://img.shields.io/badge/HF-Transformers-ffcc4d?logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers)
[![Sentence Transformers](https://img.shields.io/badge/SBERT-5.1.0-00b39f)](https://www.sbert.net/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ready-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![pytest](https://img.shields.io/badge/pytest-8.x-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Ruff](https://img.shields.io/badge/Ruff-linting-0A5A83?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/mypy-type_checking-2A6DB2?logo=python&logoColor=white)](https://mypy-lang.org/)

**`qareen`** (قرين) is Arabic for "constant companion"—an ever-present guide that influences
decisions. The project plays the same role for Large Language Models by pairing them with the right
few-shot examples at the right moment to sharpen judgment and improve downstream evaluation.

## Overview

`qareen` is a research tool for analyzing and optimizing multimodal few-shot example selection. It
extends Maximum Marginal Relevance (MMR) retrieval with a tunable alpha parameter that balances
image and text similarity, helping researchers discover the ideal mixture of modalities for their
tasks. The tool is model-agnostic: it can work with CLIP, SIGLIP, or other Hugging Face embedding
models and provides an interactive Gradio UI for exploration.

## Why it matters

- **Trustworthy evaluation:** Mitigates position bias and over-prompting issues highlighted by
  [Zheng et al. (2024)](#ref1) and [Tang et al. (2025)](#ref2).
- **Balanced retrieval:** Uses MMR to keep results both relevant and diverse, reducing redundancy in
  naive top-k similarity searches.
- **Modality control:** Lets you experiment with text–image weighting to match the needs of product
  search, relevance scoring, or quality assessment.

## Key capabilities

- Multimodal MMR retrieval with adjustable alpha for text/image weighting.
- Drop-in support for Hugging Face embedding backends (e.g., CLIP, SIGLIP).
- Gradio dashboard to visualize retrieval quality and compare model choices.
- Utilities for inspecting GPU availability before running heavier experiments.

## Installation

Install the base package:

```bash
pip install qareen
```

### GPU support

The `gpu` extra is currently a placeholder and does not install GPU-specific packages. For GPU
acceleration, install a CUDA-enabled PyTorch build from the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/) **before** installing
`qareen`. The package will warn at runtime if CUDA is missing; CPU-only usage works out of the box.

## Quickstart

```python
from qareen import check_gpu_available

if check_gpu_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Running qareen with {device} support")
```

From here you can plug your preferred Hugging Face multimodal encoder into an MMR retrieval flow and
experiment with different alpha values to balance text and image similarity.

## Development

- Install dependencies: `uv sync --all-extras`
- Run static checks: `ruff check . && mypy .`
- Run tests: `pytest`

## References

1. <a id="ref1"></a>**Zheng, L., et al.** (2024). Judging the Judges: A Systematic Study of Position
   Bias in LLM-as-a-Judge. *arXiv preprint arXiv:2406.07791*.
   [https://arxiv.org/abs/2406.07791](https://arxiv.org/abs/2406.07791)

2. <a id="ref2"></a>**Tang, Y., et al.** (2025). The Few-shot Dilemma: Over-prompting Large Language
   Models. *arXiv preprint arXiv:2509.13196*.
   [https://arxiv.org/abs/2509.13196](https://arxiv.org/abs/2509.13196)

3. <a id="ref3"></a>**Carbonell, J., & Goldstein, J.** (1998). The use of MMR, diversity-based
   reranking for reordering documents and producing summaries. *Proceedings of the 21st Annual
   International ACM SIGIR Conference on Research and Development in Information Retrieval
   (SIGIR '98)*, 335-336. [https://doi.org/10.1145/290941.291025](https://doi.org/10.1145/290941.291025)

4. <a id="ref4"></a>**Al Ghossein, M., Chen, C.-W., & Tang, J.** (2024). Shopping Queries Image
   Dataset (SQID): An Image-Enriched ESCI Dataset for Exploring Multimodal Learning in Product
   Search. Part of the Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product
   Search by Amazon.

5. <a id="ref5"></a>**Zhao, T. Z., et al.** (2021). Calibrate Before Use: Improving Few-Shot
   Performance of Language Models. *Proceedings of the International Conference on Machine Learning
   (ICML)*. [https://arxiv.org/abs/2102.09690](https://arxiv.org/abs/2102.09690)
