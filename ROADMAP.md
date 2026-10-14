# Open-source release roadmap

This page records what can currently be reproduced from the public repository and what still requires cleanup.

## Available

| Artifact | Location |
|---|---|
| ICLR 2026 paper | [Conference proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ad5b3f324b24c17cdc2f3712298c76bd-Abstract-Conference.html) |
| Structured dataset | [FreeFrank/Structured-Reasoning](https://huggingface.co/datasets/FreeFrank/Structured-Reasoning) |
| Structured SFT | [`scripts/train_sft.py`](scripts/train_sft.py) |
| Step parser | [`src/structured_reasoning/steps.py`](src/structured_reasoning/steps.py) |
| MaxFlow kernel | [`src/structured_reasoning/graph_rewards.py`](src/structured_reasoning/graph_rewards.py) |
| LCS kernel | [`src/structured_reasoning/lcs_rewards.py`](src/structured_reasoning/lcs_rewards.py) |
| Evaluation analysis | [`scripts/analyze_evaluations.py`](scripts/analyze_evaluations.py) |
| Trace analysis | [`scripts/analyze_traces.py`](scripts/analyze_traces.py) |
| Research demo | [Interactive analyzer](https://cnsdqd-dyb.github.io/Enhancing-Large-Language-Models-through-Structured-Reasoning/analyzer.html#SRA) |

## In preparation

| Artifact | Current boundary |
|---|---|
| Attention extraction | The released MaxFlow kernel accepts an aggregated step matrix; the instrumented model and layer/head aggregation pipeline are not yet public. |
| End-to-end GRPO | Reward kernels are public; the distributed launcher and trainer integration are being separated from the original experimental framework. |
| Exact experiment configs | Public SFT defaults are usable examples and are not asserted to reproduce every paper run. |
| Model checkpoints | Weights and model cards are not yet released. |

The repository intentionally excludes private machine paths, credentials, raw experiment caches, third-party source trees, and model weights. Watch the repository or use [GitHub Issues](https://github.com/cnsdqd-dyb/Enhancing-Large-Language-Models-through-Structured-Reasoning/issues) for release updates and reproducibility questions.
