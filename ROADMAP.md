# Open-source release plan

This roadmap describes the intended release scope for **Structured Reasoning for LLMs: A Unified Framework for Efficiency and Explainability** (ICLR 2026). Dates for planned artifacts have not been announced.

## Available now

| Resource | Location |
|---|---|
| Conference paper | [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ad5b3f324b24c17cdc2f3712298c76bd-Abstract-Conference.html) |
| Project homepage | [Structured Reasoning](https://cnsdqd-dyb.github.io/Enhancing-Large-Language-Models-through-Structured-Reasoning/) |
| Dataset | [FreeFrank/Structured-Reasoning](https://huggingface.co/datasets/FreeFrank/Structured-Reasoning): 516 examples, 23 cognitive step types, one train split, MIT license |
| Research demo | [Interactive reasoning analyzer](https://cnsdqd-dyb.github.io/Enhancing-Large-Language-Models-through-Structured-Reasoning/analyzer.html#SRA) |

The dataset includes reasoning text and step annotations. Attention weights and dependency graph edges are not included in the text dataset. The demo uses separate research visualization artifacts.

## Planned releases

| Stage | Intended artifacts | Status |
|---|---|---|
| Data tools and training | Annotation/validation utilities, structured SFT implementation, MaxFlow/LCS reward implementations | Planned; date to be announced |
| Reproduction and evaluation | Environment setup, experiment configurations, evaluation scripts, reproduction instructions | Planned; date to be announced |
| Model checkpoints | Weights, model cards, and inference examples | Planned; date and licenses to be announced |

These artifacts are not yet released in this repository. This roadmap describes intended scope rather than a committed release date. The dataset's MIT license applies to the dataset release; forthcoming code and model licenses will be specified separately.

## Following the release

Watch this repository for updates. Questions, reproducibility requests, and feedback can be submitted through [GitHub Issues](https://github.com/cnsdqd-dyb/Enhancing-Large-Language-Models-through-Structured-Reasoning/issues).

## Website maintenance

The full website source is maintained in `docs/` of this main repository. The former website repository is kept only for URL compatibility with the conference paper. Update the homepage and demo here when release status changes.
