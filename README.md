# qareen
*A multimodal few-shot companion that balances relevance and diversity for LLM-as-a-Judge workflows.*

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white&style=flat-square)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black&style=flat-square)](https://huggingface.co/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=flat-square)](https://www.trychroma.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063?logo=pydantic&logoColor=white&style=flat-square)](https://docs.pydantic.dev/)
![CodeRabbit Reviews](https://img.shields.io/coderabbit/prs/github/zaffnet/qareen)

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

## Demo

We demonstrate `qareen` on the Shopping Queries Image Dataset (SQID) [Al Ghossein et al. (2024)](#ref4), part of Amazon's ESCI benchmark for product search.

<!-- [Live Demo URL placeholder] -->

## Getting started

### Installation

Install the base package from PyPI:

```bash
pip install qareen
```

To set up the full development environment with linting, type-checking, and testing tools:

**Python 3.13+ Compatibility:** The package supports Python 3.13 and 3.14. Both `sentencepiece` and PyTorch now provide prebuilt cp313 wheels on PyPI. If you encounter rare environment-specific build issues, Python 3.11 or 3.12 remain stable fallback options. CI testing currently targets Python 3.11 and 3.12.

## Usage

> **Note on GPU support:** The `gpu` extra is currently a placeholder and does not install any
> GPU-specific packages. For GPU support, install a CUDA-enabled PyTorch build from the
> [official PyTorch installation guide](https://pytorch.org/get-started/locally/) **before**
> installing `qareen`. The package works with CPU-only PyTorch as well.

**Marqo Fashion Experiment:** Run `./experiments/marqo_fashion/run_experiment.sh` to compare 4 embedding models across 9 alpha values on fashion dataset.

## Configuration

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for environment variables, logging setup, and telemetry settings.

See [docs/DISTANCE_METRIC.md](docs/DISTANCE_METRIC.md) for details on the cosine distance metric used for similarity search.

## Quick Start

Run the end-to-end example (requires dataset and environment configuration):

```bash
# Set required environment variables (or use a .env config file)
export QAREEN_EMBEDDING_MODELS='["google/siglip-base-patch16-224"]'
export QAREEN_ALPHA_VALUES='[0.0, 0.5, 1.0]'
export QAREEN_ENVIRONMENT="dev"
# ... see docs/CONFIGURATION.md for all required settings

uv run python scripts/build_index.py --dataset-name <data_dir>
```

See [docs/LOCAL_DATA_GUIDE.md](docs/LOCAL_DATA_GUIDE.md) for detailed steps.

## References

1. <a id="ref1"></a>**Zheng, L., et al.** (2024). Judging the Judges: A Systematic Study of Position
   Bias in LLM-as-a-Judge. *arXiv preprint arXiv:2406.07791*.
   [https://arxiv.org/abs/2406.07791](https://arxiv.org/abs/2406.07791)
2. <a id="ref2"></a>**Tang, Y., et al.** (2025). The Few-shot Dilemma: Over-prompting Large Language
   Models. *arXiv preprint arXiv:2509.13196*.
   [https://arxiv.org/abs/2509.13196](https://arxiv.org/abs/2509.13196)
3. <a id="ref3"></a>**Carbonell, J., & Goldstein, J.** (1998). The use of MMR, diversity-based
   reranking for reordering documents and producing summaries. *Proceedings of SIGIR '98*, 335-336.
   [https://doi.org/10.1145/290941.291025](https://doi.org/10.1145/290941.291025)
4. <a id="ref4"></a>**Al Ghossein, M., Chen, C.-W., & Tang, J.** (2024). Shopping Queries Image
   Dataset (SQID): An Image-Enriched ESCI Dataset for Exploring Multimodal Learning in Product
   Search. Part of the Shopping Queries Dataset by Amazon.
   [Paper Link](https://www.amazon.science/publications/shopping-queries-image-dataset-sqid-an-image-enriched-esci-dataset-for-exploring-multimodal-learning-in-product-search)
5. <a id="ref5"></a>**Zhao, T. Z., et al.** (2021). Calibrate Before Use: Improving Few-Shot
   Performance of Language Models. *Proceedings of the International Conference on Machine
   Learning (ICML)*. [https://arxiv.org/abs/2102.09690](https://arxiv.org/abs/2102.09690)
