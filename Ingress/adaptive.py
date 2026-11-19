from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PruneDecision:
    prune: bool
    bound: float


def frobenius_product_bound(a_block: np.ndarray, b_block: np.ndarray) -> float:
    return float(np.linalg.norm(a_block, ord="fro") * np.linalg.norm(b_block, ord="fro"))


def should_prune(
    a_block: np.ndarray,
    b_block: np.ndarray,
    epsilon: float = 0.0,
) -> PruneDecision:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    if not np.any(a_block) or not np.any(b_block):
        return PruneDecision(prune=True, bound=0.0)

    if epsilon == 0.0:
        return PruneDecision(prune=False, bound=frobenius_product_bound(a_block, b_block))

    bound = frobenius_product_bound(a_block, b_block)
    return PruneDecision(prune=bound <= epsilon, bound=bound)


@dataclass(frozen=True)
class LowRankFactors:
    x: np.ndarray
    y: np.ndarray
    rank: int
    relative_error: float


def truncated_svd_factors(
    block: np.ndarray,
    tolerance: float = 1e-6,
    max_rank: int | None = None,
) -> LowRankFactors | None:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    block = np.asarray(block, dtype=np.float64)
    norm = float(np.linalg.norm(block, ord="fro"))
    if norm == 0.0:
        return LowRankFactors(
            x=np.zeros((block.shape[0], 0), dtype=np.float64),
            y=np.zeros((block.shape[1], 0), dtype=np.float64),
            rank=0,
            relative_error=0.0,
        )

    u, s, vt = np.linalg.svd(block, full_matrices=False)
    limit = len(s) if max_rank is None else min(max_rank, len(s))

    for rank in range(1, limit + 1):
        approx = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
        rel_error = float(np.linalg.norm(block - approx, ord="fro") / norm)
        if rel_error <= tolerance:
            x = np.ascontiguousarray(u[:, :rank] * s[:rank])
            y = np.ascontiguousarray(vt[:rank, :].T)
            return LowRankFactors(x=x, y=y, rank=rank, relative_error=rel_error)

    return None


def factor_payload_is_smaller(block: np.ndarray, factors: LowRankFactors) -> bool:
    dense_values = block.size
    factor_values = factors.x.size + factors.y.size
    return factor_values < dense_values
