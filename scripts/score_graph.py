#!/usr/bin/env python3
"""Compute the paper's MaxFlow graph reward from a JSON attention matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from structured_reasoning import maxflow_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path, help="JSON file containing an N x N matrix")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    result = maxflow_reward(matrix)
    print(json.dumps({
        "reward": result.reward,
        "max_flow": result.max_flow,
        "importance": result.importance,
    }, indent=2))


if __name__ == "__main__":
    main()
