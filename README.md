# qareen

**qareen** (قرين) is Arabic for "constant companion"—an ever-present guide that influences
choices. This library plays the same role for Large Language Models: it selects the right
few-shot multimodal examples at the right time so your judges stay reliable.

## Technology Stickers

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-fdd835?logo=huggingface&logoColor=black)
![SentenceTransformers](https://img.shields.io/badge/Sentence%20Transformers-5.1.0-1f4b99)
![Gradio](https://img.shields.io/badge/Gradio-4.0-ff6f61)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20ready-ee4c2c?logo=pytorch&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-8.0-0A9EDC?logo=pytest&logoColor=white)
![ruff](https://img.shields.io/badge/Ruff-linting-cf3c3c)
![mypy](https://img.shields.io/badge/mypy-type%20checks-20588f)

## Why qareen?

Few-shot example selection drives LLM-as-a-Judge performance, yet optimal strategies remain
unclear. Position bias [Zheng et al. (2024)](#ref1) and over-prompting [Tang et al.
(2025)](#ref2) both degrade quality, while naive similarity search returns redundant examples.
`qareen` counters these pitfalls by balancing relevance and diversity across text and image
modalities.

## What it does

- **Multimodal MMR retrieval:** Extends Maximum Marginal Relevance with a tunable alpha to
  control text vs. image similarity.
- **Model-agnostic embeddings:** Works with any Hugging Face multimodal encoder (CLIP, SIGLIP,
  domain-specific variants).
- **Interactive experimentation:** Gradio UI lets you adjust alpha, swap embedding models, and
  visualize retrieved examples in real time.
- **Dataset insights:** Designed for product-search benchmarks like the Shopping Queries Image
  Dataset (SQID), highlighting how modality weighting changes retrieval quality.

## Architecture at a glance

1. Ingest multimodal data and build embeddings with your preferred Hugging Face model.
2. Store vectors and metadata, then apply MMR to re-rank candidates for relevance and diversity.
3. Explore results in the Gradio dashboard, iterating on alpha and model selection to find the
   sweet spot for your task.

## Installation

```bash
pip install qareen
```

> **GPU note:** The `gpu` extra is a placeholder. For GPU acceleration, first install a
> CUDA-enabled PyTorch build from the [official guide](https://pytorch.org/get-started/locally/)
> before installing `qareen`. The base package runs on CPU-only PyTorch as well.

## Quickstart

1. Prepare your dataset (e.g., SQID) with image and text fields.
2. Build embeddings and an index:
   ```bash
   ./scripts/build_index.sh <path-to-data>
   ```
3. Launch the Gradio interface:
   ```bash
   ./scripts/run_gradio.sh
   ```
4. Tune the alpha slider and swap models to observe how retrieval quality changes.

## Development setup

```bash
uv sync --all-extras
```

- Linting: `uv run ruff check .`
- Type checks: `uv run mypy .`
- Tests: `uv run pytest`

## References

1. <a id="ref1"></a>**Zheng, L., et al.** (2024). Judging the Judges: A Systematic Study of
   Position Bias in LLM-as-a-Judge. *arXiv preprint arXiv:2406.07791*. [https://arxiv.org/abs/2406.07791](https://arxiv.org/abs/2406.07791)
2. <a id="ref2"></a>**Tang, Y., et al.** (2025). The Few-shot Dilemma: Over-prompting Large
   Language Models. *arXiv preprint arXiv:2509.13196*. [https://arxiv.org/abs/2509.13196](https://arxiv.org/abs/2509.13196)
3. <a id="ref3"></a>**Carbonell, J., & Goldstein, J.** (1998). The use of MMR, diversity-based
   reranking for reordering documents and producing summaries. *Proceedings of the 21st Annual
   International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR
   '98)*, 335-336. [https://doi.org/10.1145/290941.291025](https://doi.org/10.1145/290941.291025)
4. <a id="ref4"></a>**Al Ghossein, M., Chen, C.-W., & Tang, J.** (2024). Shopping Queries Image
   Dataset (SQID): An Image-Enriched ESCI Dataset for Exploring Multimodal Learning in Product
   Search. Part of the Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving
   Product Search by Amazon.
5. <a id="ref5"></a>**Zhao, T. Z., et al.** (2021). Calibrate Before Use: Improving Few-Shot
   Performance of Language Models. *Proceedings of the International Conference on Machine
   Learning (ICML)*. [https://arxiv.org/abs/2102.09690](https://arxiv.org/abs/2102.09690)
