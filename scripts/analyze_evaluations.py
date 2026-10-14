#!/usr/bin/env python3
"""Aggregate LightEval JSON outputs across seeds and decoding settings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev


def aggregate(directory: Path) -> list[dict]:
    values: dict[tuple, list[float]] = defaultdict(list)
    for path in sorted(directory.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        config = payload.get("config_general", {}).get("generation_parameters", {})
        max_tokens = config.get("max_new_tokens")
        temperature = config.get("temperature")
        for task, metrics in payload.get("results", {}).items():
            if task == "all" or not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                if metric.endswith("_stderr") or not isinstance(value, (int, float)):
                    continue
                values[(max_tokens, temperature, task, metric)].append(float(value))
    return [
        {
            "max_new_tokens": key[0],
            "temperature": key[1],
            "task": key[2],
            "metric": key[3],
            "mean": fmean(samples),
            "std": pstdev(samples),
            "runs": len(samples),
        }
        for key, samples in sorted(values.items(), key=lambda item: str(item[0]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation_summary.csv"))
    args = parser.parse_args()
    rows = aggregate(args.input)
    if not rows:
        raise SystemExit(f"no LightEval metrics found under {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} aggregated metrics to {args.output}")


if __name__ == "__main__":
    main()
