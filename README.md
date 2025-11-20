# qareen

*A multimodal few-shot companion that balances relevance and diversity for LLM-as-a-Judge workflows.*

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white&style=flat-square)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black&style=flat-square)](https://huggingface.co/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-1C3C3C?logo=langchain&logoColor=white&style=flat-square)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=flat-square)](https://www.trychroma.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063?logo=pydantic&logoColor=white&style=flat-square)](https://docs.pydantic.dev/)
[![CodeRabbit](https://img.shields.io/badge/CodeRabbit-AI%20Review-blue?style=flat-square)](https://coderabbit.ai/)

## Overview

**`qareen`** (قرين) means "constant companion"—a guide that subtly shapes decisions. The
project plays the same role for Large Language Models: it supplies the right few-shot examples
at the right moment, guiding model judgments for multimodal tasks that mix text and images.

## Why it matters

Few-shot selection significantly influences LLM-as-a-Judge quality. Position bias, redundant
examples, and modality imbalance can all distort evaluations. `qareen` addresses these pitfalls
by extending Maximum Marginal Relevance (MMR) to multimodal retrieval with a tunable `alpha`
parameter that controls text–image weighting.

## Key features

- **Multimodal MMR retrieval:** Balance relevance and diversity across text and image signals.
- **Model flexibility:** Swap between CLIP, SIGLIP, or other Hugging Face embeddings via
  `transformers` and `sentence-transformers`.
- **Interactive exploration:** Adjust modality weights live through a Gradio UI to see how
  examples shift.
- **GPU-aware runtime:** Detects CUDA availability and guides you to install a compatible
  PyTorch build when acceleration is possible.
- **Type-safe development:** Ruff + mypy + pytest keep the codebase linted, typed, and tested.

## Demo

We showcase `qareen` on the Shopping Queries Image Dataset (SQID) [Al Ghossein et al. (2024)](#ref4),
part of Amazon's ESCI benchmark for product search.

<!-- ![Demo GIF placeholder] -->

<!-- [Live Demo URL placeholder] -->

## Getting started

### Installation

Install the base package from PyPI:

```bash
pip install qareen
```

To set up the full development environment with linting, type-checking, and testing tools:

```bash
uv sync --all-extras
```

> **Note on GPU support:** The `gpu` extra is currently a placeholder and does not install any
> GPU-specific packages. For GPU support, install a CUDA-enabled PyTorch build from the
> [official PyTorch installation guide](https://pytorch.org/get-started/locally/) **before**
> installing `qareen`. The package works with CPU-only PyTorch as well.

### Quickstart

Run the end-to-end example (requires SQID dataset – see docs/QUICKSTART.md for data preparation):

```bash
./scripts/build_index.sh <data_dir>
./scripts/run_gradio.sh
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed steps.

## References

1. <a id="ref1"></a>**Zheng, L., et al.** (2024). Judging the Judges: A Systematic Study of Position
   Bias in LLM-as-a-Judge. *arXiv preprint arXiv:2406.07791*.
   [https://arxiv.org/abs/2406.07791](https://arxiv.org/abs/2406.07791)
2. <a id="ref2"></a>**Tang, Y., et al.** (2025). The Few-shot Dilemma: Over-prompting Large Language
   Models. *arXiv preprint arXiv:2509.13196*.
   [https://arxiv.org/abs/2509.13196](https://arxiv.org/abs/2509.13196)
3. <a id="ref3"></a>**Carbonell, J., & Goldstein, J.** (1998). The use of MMR, diversity-based
   reranking for reordering documents and producing summaries. *SIGIR '98*.
   [https://doi.org/10.1145/290941.291025](https://doi.org/10.1145/290941.291025)
4. <a id="ref4"></a>**Al Ghossein, M., Chen, C.-W., & Tang, J.** (2024). Shopping Queries Image
   Dataset (SQID): An Image-Enriched ESCI Dataset for Exploring Multimodal Learning in Product
   Search. Part of the Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving
   Product Search by Amazon.
5. <a id="ref5"></a>**Zhao, T. Z., et al.** (2021). Calibrate Before Use: Improving Few-Shot
   Performance of Language Models. *Proceedings of the International Conference on Machine
   Learning (ICML)*. [https://arxiv.org/abs/2102.09690](https://arxiv.org/abs/2102.09690)
