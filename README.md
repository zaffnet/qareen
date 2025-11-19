# qareen

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white&style=flat-square)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black&style=flat-square)](https://huggingface.co/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-1C3C3C?logo=langchain&logoColor=white&style=flat-square)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=flat-square)](https://www.trychroma.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063?logo=pydantic&logoColor=white&style=flat-square)](https://docs.pydantic.dev/)
[![CodeRabbit](https://img.shields.io/badge/CodeRabbit-AI%20Review-blue?style=flat-square)](https://coderabbit.ai/)

**`qareen`** (قرين) is Arabic for "constant companion"—an ever-present guide that influences decisions. In Islamic tradition, a qareen is a spiritual companion assigned to each person, accompanying them throughout life and shaping their choices through subtle guidance.

This project is named `qareen` because it serves the same role for Large Language Models: a constant companion that provides the right few-shot examples at the right time, guiding the LLM's judgments and enhancing its decision-making. Just as a qareen influences a person's path, `qareen` influences an LLM's performance by selecting the most relevant multimodal examples from your dataset.

## The Problem

Few-shot example selection impacts LLM-as-a-Judge performance, yet optimal selection strategies remain poorly understood. [Zheng et al. (2024)](#ref1) demonstrate that LLM judges systematically favor responses based on position rather than quality, undermining evaluation reliability. [Tang et al. (2025)](#ref2) show that over-prompting with too many examples degrades performance, while naive top-k similarity retrieval returns redundant examples that fail to clarify task boundaries. The broader problem can be thought of as striking a balance between providing relevant few-shot examples (to allow in-context learning) while maintaining diversity (to avoid overfitting).

For multimodal tasks, the challenge is twofold: you must balance relevance and diversity, and decide how much weight to assign to each modality (text or image). This is particularly important when LLMs evaluate product descriptions, score image–text relevance, or assess quality, since poor example selection can directly impact LLM-as-a-Judge.


## Our Solution

`qareen` is a tool for analyzing and optimizing multimodal few-shot example selection. It extends Maximum Marginal Relevance to multimodal retrieval by introducing a tunable alpha parameter that controls the relative weight of image versus text similarity, enabling researchers to explore the optimal balance between modalities for a given task and dataset.

The tool works with any multimodal embedding model from Hugging Face—CLIP, SIGLIP, or domain-specific variants—allowing comparison of how different architectures handle modality weighting. Through an interactive Gradio interface, users can adjust the alpha parameter in real-time, switch between embedding models, and observe how these choices affect retrieved examples. `qareen` implements MMR-based retrieval that balances relevance and diversity, addressing the redundancy problem that plagues naive k-NN approaches.

## Demo

We demonstrate `qareen` on the Shopping Queries Image Dataset (SQID) [Al Ghossein et al. (2024)](#ref4), part of Amazon's ESCI benchmark for product search. 

![Demo GIF placeholder]

[Live Demo URL placeholder]

## Installation

Install the base package:

```bash
pip install qareen
```

**Note on GPU support:** The `gpu` extra is currently a placeholder and does not install any GPU-specific packages. For GPU support, you must install a CUDA-enabled PyTorch build from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) **before** installing `qareen`. The package will automatically check at runtime and warn if CUDA is not available. The base package will work with CPU-only PyTorch (installed automatically via dependencies).

## Usage

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for full steps; e.g., `./scripts/build_index.sh <data> && ./scripts/run_gradio.sh`

## References

1. <a id="ref1"></a>**Zheng, L., et al.** (2024). Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge. *arXiv preprint arXiv:2406.07791*. [https://arxiv.org/abs/2406.07791](https://arxiv.org/abs/2406.07791)

2. <a id="ref2"></a>**Tang, Y., et al.** (2025). The Few-shot Dilemma: Over-prompting Large Language Models. *arXiv preprint arXiv:2509.13196*. [https://arxiv.org/abs/2509.13196](https://arxiv.org/abs/2509.13196)

3. <a id="ref3"></a>**Carbonell, J., & Goldstein, J.** (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '98)*, 335-336. [https://doi.org/10.1145/290941.291025](https://doi.org/10.1145/290941.291025)

4. <a id="ref4"></a>**Al Ghossein, M., Chen, C.-W., & Tang, J.** (2024). Shopping Queries Image Dataset (SQID): An Image-Enriched ESCI Dataset for Exploring Multimodal Learning in Product Search. Part of the Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search by Amazon.

5. <a id="ref5"></a>**Zhao, T. Z., et al.** (2021). Calibrate Before Use: Improving Few-Shot Performance of Language Models. *Proceedings of the International Conference on Machine Learning (ICML)*. [https://arxiv.org/abs/2102.09690](https://arxiv.org/abs/2102.09690)
