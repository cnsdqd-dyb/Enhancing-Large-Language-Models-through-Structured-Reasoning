"""LCS reward for comparing multiple structured reasoning trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Step:
    tag: str
    content: str


def _jaccard(left: str, right: str) -> float:
    left_words, right_words = set(left.lower().split()), set(right.lower().split())
    union = left_words | right_words
    return len(left_words & right_words) / len(union) if union else 0.0


def lcs_step_weights(
    left: Sequence[Step], right: Sequence[Step], *, similarity_threshold: float = 0.6
) -> list[float]:
    """Return length-aware weights for the fuzzy LCS matched in ``left``."""

    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in [0, 1]")
    if not left or not right:
        return [0.0] * len(left)

    rows, cols = len(left), len(right)
    matches = {
        (i, j)
        for i, a in enumerate(left)
        for j, b in enumerate(right)
        if a.tag == b.tag or _jaccard(a.content, b.content) >= similarity_threshold
    }
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1
                if (i - 1, j - 1) in matches
                else max(table[i - 1][j], table[i][j - 1])
            )

    aligned: dict[int, int] = {}
    i, j = rows, cols
    while i and j:
        if (i - 1, j - 1) in matches and table[i][j] == table[i - 1][j - 1] + 1:
            aligned[i - 1] = j - 1
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1

    weights = [0.0] * rows
    for left_index, right_index in aligned.items():
        left_length = len(left[left_index].content)
        right_length = len(right[right_index].content)
        if left_length and right_length:
            weights[left_index] = (
                right_length / left_length / 2
                if left_length > right_length
                else 1 - left_length / right_length / 2
            )
    return weights


def _weighted_overlap(left: Sequence[Step], right: Sequence[Step]) -> float:
    weights = lcs_step_weights(left, right)
    total = sum(len(step.content) for step in left)
    return (
        sum(weight * len(step.content) for weight, step in zip(weights, left)) / total
        if total
        else 0.5
    )


def _normalize(values: list[float], indices: Iterable[int], low: float, high: float) -> None:
    chosen = list(indices)
    if not chosen:
        return
    minimum, maximum = min(values[i] for i in chosen), max(values[i] for i in chosen)
    if maximum == minimum:
        midpoint = (low + high) / 2
        for i in chosen:
            values[i] = midpoint
        return
    for i in chosen:
        values[i] = low + (values[i] - minimum) / (maximum - minimum) * (high - low)


def group_lcs_rewards(traces: Sequence[Sequence[Step]], correct: Sequence[bool]) -> list[float]:
    """Compute group-relative LCS rewards for sampled responses to one prompt."""

    if len(traces) != len(correct):
        raise ValueError("traces and correct must have the same length")
    count = len(traces)
    if count <= 1:
        return [1.0 if flag else -1.0 for flag in correct]

    rewards = [0.0] * count
    for i, trace in enumerate(traces):
        pair_scores: list[float] = []
        for j, other in enumerate(traces):
            if i == j:
                continue
            common = _weighted_overlap(trace, other)
            if correct[i] and correct[j]:
                score = common
            elif not correct[i] and not correct[j]:
                score = -common
            elif correct[i]:
                score = 1 - common
            else:
                score = -1 + common
            pair_scores.append(score)
        rewards[i] = sum(pair_scores) / len(pair_scores)

    _normalize(rewards, (i for i, flag in enumerate(correct) if flag), 0.0, 1.0)
    _normalize(rewards, (i for i, flag in enumerate(correct) if not flag), -1.5, -0.5)
    return rewards
