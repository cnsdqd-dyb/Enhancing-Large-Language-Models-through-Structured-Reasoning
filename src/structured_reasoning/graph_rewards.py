"""MaxFlow reward used for single structured-reasoning graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class MaxFlowResult:
    """The graph-level reward and per-step leave-one-out contributions."""

    reward: float
    max_flow: float
    importance: tuple[float, ...]


def _as_capacity_matrix(values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("attention matrix must be square")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("attention capacities must be finite and non-negative")
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    return matrix


def sparsify(matrix: np.ndarray, keep_ratio: float = 0.2) -> np.ndarray:
    """Keep the largest positive capacities, matching the experiment code."""

    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    positive = matrix[matrix > 0]
    if positive.size == 0 or keep_ratio == 1:
        return matrix.copy()
    threshold = np.percentile(positive, (1 - keep_ratio) * 100)
    result = matrix.copy()
    result[result < threshold] = 0.0
    return result


def _graph(matrix: np.ndarray, removed: int | None = None) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(i for i in range(len(matrix)) if i != removed)
    for source, target in zip(*np.nonzero(matrix)):
        if source != removed and target != removed:
            graph.add_edge(
                int(source), int(target), capacity=float(matrix[source, target])
            )
    return graph


def _flow(graph: nx.DiGraph, source: int, sink: int) -> float:
    if source not in graph or sink not in graph:
        return 0.0
    try:
        return float(nx.maximum_flow_value(graph, source, sink))
    except (nx.NetworkXError, nx.NetworkXUnbounded):
        return 0.0


def maxflow_reward(
    attention: Sequence[Sequence[float]],
    *,
    sparsify_over: int = 200,
    keep_ratio: float | None = None,
    balance_parts: int = 4,
) -> MaxFlowResult:
    """Score a reasoning DAG from its step-to-step attention capacities.

    Each intermediate node receives its leave-one-out contribution: the drop
    in source-to-answer maximum flow when that step is removed. The graph
    reward measures how much contribution remains outside the top quarter,
    rewarding balanced paths and discouraging redundant steps.
    """

    matrix = _as_capacity_matrix(attention)
    count = len(matrix)
    if count == 0:
        return MaxFlowResult(0.0, 0.0, ())
    if count == 1:
        return MaxFlowResult(1.0, 0.0, (1.0,))
    if balance_parts < 2:
        raise ValueError("balance_parts must be at least 2")

    if count > sparsify_over:
        ratio = keep_ratio
        if ratio is None:
            ratio = 0.30 if count <= 500 else 0.20 if count <= 1000 else 0.15 if count <= 2000 else 0.10
        matrix = sparsify(matrix, ratio)

    source, sink = 0, count - 1
    base_graph = _graph(matrix)
    base_flow = _flow(base_graph, source, sink)
    contributions = np.zeros(count, dtype=float)
    for node in range(1, sink):
        contributions[node] = max(
            0.0, base_flow - _flow(_graph(matrix, removed=node), source, sink)
        )

    endpoint_value = max(float(contributions.max()), base_flow, 1e-12) * 1.01
    contributions[source] = endpoint_value
    contributions[sink] = endpoint_value
    contributions /= contributions.max()

    ordered = np.sort(contributions)[::-1]
    head = max(1, count // balance_parts)
    total = float(ordered.sum())
    reward = float(ordered[head:].sum() / total * balance_parts) if total else 0.0
    return MaxFlowResult(reward, base_flow, tuple(float(x) for x in contributions))
