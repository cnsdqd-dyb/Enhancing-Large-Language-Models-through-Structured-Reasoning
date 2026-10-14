#!/usr/bin/env python3
"""Validate and export Structured Reasoning examples as JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


REQUIRED_FIELDS = ("problem", "reasoning", "content")


def serialize(example: dict) -> dict:
    missing = [name for name in REQUIRED_FIELDS if not example.get(name)]
    if missing:
        raise ValueError(f"example is missing required fields: {', '.join(missing)}")
    return {
        "messages": [
            {"role": "user", "content": example["problem"]},
            {
                "role": "assistant",
                "content": f"<think>\n{example['reasoning']}\n</think>\n{example['content']}",
            },
        ],
        "problem_id": example.get("problem_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FreeFrank/Structured-Reasoning")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("data/structured_reasoning.jsonl"))
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for index, example in enumerate(dataset):
            try:
                record = serialize(example)
            except ValueError as error:
                raise ValueError(f"invalid example at index {index}: {error}") from error
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} examples to {args.output}")


if __name__ == "__main__":
    main()
