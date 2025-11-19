# qareen

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Interface-FF6B2C?logo=gradio&logoColor=white)
![Hugging%20Face%20Transformers](https://img.shields.io/badge/Transformers-🤗-FFD21E?logo=huggingface&logoColor=black)
![SentenceTransformers](https://img.shields.io/badge/Sentence%20Transformers-Embeddings-0D1117?logo=semanticweb&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Frames-150458?logo=pandas&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-Tests-0A9EDC?logo=pytest&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-Linting-000000?logo=ruff&logoColor=white)
![mypy](https://img.shields.io/badge/mypy-Static%20Typing-2A6DB2?logo=python&logoColor=white)
![pre-commit](https://img.shields.io/badge/pre--commit-Hooks-FAB040?logo=pre-commit&logoColor=white)

**`qareen`** (قرين) is Arabic for "constant companion"—an ever-present guide that influences
decisions. In Islamic tradition, a qareen is a spiritual companion assigned to each person,
accompanying them throughout life and shaping their choices through subtle guidance.

This project is named `qareen` because it serves the same role for Large Language Models: a constant
companion that provides the right few-shot examples at the right time, guiding the LLM's judgments
and enhancing its decision-making. Just as a qareen influences a person's path, `qareen` influences
an LLM's performance by selecting the most relevant multimodal examples from your dataset.

## Table of Contents

- [Problem](#problem)
- [Solution](#solution)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Development](#development)
- [References](#references)

## Problem

Few-shot example selection impacts LLM-as-a-Judge performance, yet optimal selection strategies
remain poorly understood. [Zheng et al. (2024)](#ref1) demonstrate that LLM judges systematically
favor responses based on position rather than quality, undermining evaluation reliability. [Tang
et al. (2025)](#ref2) show that over-prompting with too many examples degrades performance, while
naive top-k similarity retrieval returns redundant examples that fail to clarify task boundaries.
The broader problem can be thought of as striking a balance between providing relevant few-shot
examples (to allow in-context learning) while maintaining diversity (to avoid overfitting).

For multimodal tasks, the challenge is twofold: you must balance relevance and diversity, and decide
how much weight to assign to each modality (text or image). This is particularly important when LLMs
evaluate product descriptions, score image–text relevance, or assess quality, since poor example
selection can directly impact LLM-as-a-Judge.

## Solution

`qareen` is a tool for analyzing and optimizing multimodal few-shot example selection. It extends
Maximum Marginal Relevance to multimodal retrieval by introducing a tunable alpha parameter that
controls the relative weight of image versus text similarity, enabling researchers to explore the
optimal balance between modalities for a given task and dataset.

The tool works with any multimodal embedding model from Hugging Face—CLIP, SIGLIP, or
domain-specific variants—allowing comparison of how different architectures handle modality
weighting. Through an interactive Gradio interface, users can adjust the alpha parameter in
real-time, switch between embedding models, and observe how these choices affect retrieved
examples. `qareen` implements MMR-based retrieval that balances relevance and diversity, addressing
the redundancy problem that plagues naive k-NN approaches.

## Features

- 🔍 **Multimodal MMR retrieval:** Balance text and image similarity with a tunable alpha parameter
  to surface diverse, high-signal few-shot examples.
- 🧠 **Model flexibility:** Plug in Hugging Face models (e.g., CLIP, SIGLIP) via Transformers and
  SentenceTransformers without changing core logic.
- 🖥️ **Interactive exploration:** Use the Gradio UI to experiment with modality weights and model
  choices, observing how retrieval quality shifts.
- 📊 **Transparent analytics:** Analyze retrieved examples with pandas- and numpy-powered data views
  to validate diversity and coverage.
- 🧪 **Quality gates:** Pytest, Ruff, mypy, and pre-commit hooks keep the codebase reliable and
  consistent.

## Architecture

`qareen` follows a Pydantic-first, modular structure to keep datasets, indexing strategies, and
configuration layers cleanly separated.

- **Dataset loaders:** Structured ingestion and validation for multimodal records.
- **Indexing layer:** Maximum Marginal Relevance retrieval that mixes cosine similarity across text
  and image embeddings.
- **Interactive app:** A Gradio front end for adjusting modality weights, model selection, and
  previewing ranked results.

## Installation

Install the base package:

```bash
pip install qareen
```

**Note on GPU support:** The `gpu` extra is currently a placeholder and does not install any
GPU-specific packages. For GPU support, you must install a CUDA-enabled PyTorch build from the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/) **before**
installing `qareen`. The base package will work with CPU-only PyTorch (installed automatically via
dependencies), and the package will warn if CUDA is not available.

## Quickstart

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for complete steps; a typical flow is:

```bash
./scripts/build_index.sh <path-to-data>
./scripts/run_gradio.sh
```

Use the Gradio interface to adjust the alpha parameter, swap embedding models, and review how the
retrieved few-shot examples evolve.

## Development

Set up a development environment and run quality checks:

```bash
uv sync --all-extras
uv run ruff check .
uv run mypy .
uv run pytest
```

Pre-commit hooks enforce formatting and lint rules locally:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## References

1. <a id="ref1"></a>**Zheng, L., et al.** (2024). Judging the Judges: A Systematic Study of Position
   Bias in LLM-as-a-Judge. *arXiv preprint arXiv:2406.07791*.
   [https://arxiv.org/abs/2406.07791](https://arxiv.org/abs/2406.07791)
2. <a id="ref2"></a>**Tang, Y., et al.** (2025). The Few-shot Dilemma: Over-prompting Large Language
   Models. *arXiv preprint arXiv:2509.13196*.
   [https://arxiv.org/abs/2509.13196](https://arxiv.org/abs/2509.13196)
3. <a id="ref3"></a>**Carbonell, J., & Goldstein, J.** (1998). The use of MMR, diversity-based
   reranking for reordering documents and producing summaries. *Proceedings of the 21st Annual
   International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR
   '98)*, 335-336. [https://doi.org/10.1145/290941.291025](https://doi.org/10.1145/290941.291025)
4. <a id="ref4"></a>**Al Ghossein, M., Chen, C.-W., & Tang, J.** (2024). Shopping Queries Image
   Dataset (SQID): An Image-Enriched ESCI Dataset for Exploring Multimodal Learning in Product
   Search. Part of the Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product
   Search by Amazon.
5. <a id="ref5"></a>**Zhao, T. Z., et al.** (2021). Calibrate Before Use: Improving Few-Shot
   Performance of Language Models. *Proceedings of the International Conference on Machine Learning
   (ICML)*. [https://arxiv.org/abs/2102.09690](https://arxiv.org/abs/2102.09690)
