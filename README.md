# Structured Reasoning

**Structured Reasoning for LLMs: A Unified Framework for Efficiency and Explainability**  
Yubo Dong, Hehe Fan, Linchao Zhu, and Yi Yang · ICLR 2026

[Paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ad5b3f324b24c17cdc2f3712298c76bd-Abstract-Conference.html) · [Project page](https://cnsdqd-dyb.github.io/structured-reasoning/) · [Dataset on Hugging Face](https://huggingface.co/datasets/FreeFrank/Structured-Reasoning)

This is the project repository linked from the [original arXiv paper](https://arxiv.org/abs/2506.20241). The ICLR 2026 paper links to the [project website](https://cnsdqd-dyb.github.io/structured-reasoning/), whose source is maintained in the separate [website repository](https://github.com/cnsdqd-dyb/structured-reasoning).

The framework organizes reasoning into cognitive steps to study reasoning efficiency and explainability. The dataset is publicly available below; training code and model checkpoints have not yet been released in this repository.

## Dataset

[**FreeFrank/Structured-Reasoning**](https://huggingface.co/datasets/FreeFrank/Structured-Reasoning) is publicly available under the **MIT** license. It contains **516 examples** in one `train` split, with reasoning segmented using **23 cognitive step types**. Parquet and JSONL versions are provided.

```python
from datasets import load_dataset

dataset = load_dataset("FreeFrank/Structured-Reasoning", split="train")
example = dataset[0]
print(example["problem"])
print(example["reasoning"])
print(example["answer"])
```

| Field | Description |
|---|---|
| `problem_id` | Stable example identifier |
| `problem` | Problem statement |
| `reasoning` | Reasoning with paired cognitive step tags |
| `answer` | Answer target |
| `content` | Final response |
| `steps` | Ordered steps with `step_id`, `type`, and `text` |

For supervised training, use `problem` as the user prompt and construct the assistant target as follows:

```python
assistant_target = (
    "<think>\n" + example["reasoning"]
    + "\n</think>\n" + example["content"]
)
```

Ensure your chat template retains the reasoning and your context length accommodates the full target. With the checked DeepSeek-R1-Distill-Qwen-7B tokenizer, the longest serialized example has 26,698 tokens; a 32,768-token context accommodates all examples. Recheck lengths with your own tokenizer and template.

The release includes editorial curation and step annotations. It is not asserted to be the exact dataset used for the paper's reported experiments, and no independent corpus-wide correctness estimate is reported. Question provenance, source attribution, license terms, and further usage details are available in the [dataset card](https://huggingface.co/datasets/FreeFrank/Structured-Reasoning). Step annotations describe reasoning spans; they do not provide attention weights or dependency graph edges.

## Citation

```bibtex
@inproceedings{dong2026structuredreasoning,
  title = {Structured Reasoning for LLMs: A Unified Framework for Efficiency and Explainability},
  author = {Dong, Yubo and Fan, Hehe and Zhu, Linchao and Yang, Yi},
  booktitle = {International Conference on Learning Representations},
  year = {2026},
  url = {https://proceedings.iclr.cc/paper_files/paper/2026/hash/ad5b3f324b24c17cdc2f3712298c76bd-Abstract-Conference.html}
}
```
