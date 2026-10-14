#!/usr/bin/env python3
"""Summarize tag usage and reasoning efficiency from JSONL traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

from structured_reasoning import extract_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--reasoning-field", default="reasoning")
    parser.add_argument("--correct-field", default="correct")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit("input contains no records")
    tags: Counter[str] = Counter()
    group_steps: dict[bool, list[int]] = {True: [], False: []}
    group_chars: dict[bool, list[int]] = {True: [], False: []}
    malformed = 0
    for record in records:
        reasoning = str(record.get(args.reasoning_field, ""))
        steps = extract_steps(reasoning)
        if reasoning and not steps:
            malformed += 1
        correct = bool(record.get(args.correct_field, False))
        tags.update(step.tag for step in steps)
        group_steps[correct].append(len(steps))
        group_chars[correct].append(sum(len(step.content) for step in steps))

    def mean(values: list[int]) -> float:
        return fmean(values) if values else 0.0

    report = {
        "records": len(records),
        "malformed_traces": malformed,
        "top_tags": tags.most_common(23),
        "correct": {
            "count": len(group_steps[True]),
            "mean_steps": mean(group_steps[True]),
            "mean_reasoning_characters": mean(group_chars[True]),
        },
        "incorrect": {
            "count": len(group_steps[False]),
            "mean_steps": mean(group_steps[False]),
            "mean_reasoning_characters": mean(group_chars[False]),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
