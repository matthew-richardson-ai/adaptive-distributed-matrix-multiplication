from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class SerialResult:
    matrix: np.ndarray
    elapsed_seconds: float


def validate_multiplication_shapes(a: np.ndarray, b: np.ndarray) -> None:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both inputs must be 2-D matrices.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Incompatible matrix shapes for multiplication: {a.shape} and {b.shape}."
        )


def multiply_serial(a: np.ndarray, b: np.ndarray) -> SerialResult:
    validate_multiplication_shapes(a, b)
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)

    start = time.perf_counter()
    c = a64 @ b64
    elapsed = time.perf_counter() - start
    return SerialResult(matrix=c, elapsed_seconds=elapsed)


def validate_result(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> bool:
    return bool(np.allclose(actual, expected, rtol=rtol, atol=atol))


@dataclass(frozen=True)
class BlockedSerialResult:
    matrix: np.ndarray
    elapsed_seconds: float
    block_size: int


def multiply_blocked_serial(
    a: np.ndarray,
    b: np.ndarray,
    block_size: int,
) -> BlockedSerialResult:
    """Single-threaded blocked baseline using the same block kernel shape as workers."""
    validate_multiplication_shapes(a, b)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    m, n = a64.shape
    _, p = b64.shape
    c = np.zeros((m, p), dtype=np.float64)

    start = time.perf_counter()
    with threadpool_limits(limits=1):
        for i in range(0, m, block_size):
            i_end = min(i + block_size, m)
            for j in range(0, p, block_size):
                j_end = min(j + block_size, p)
                for k in range(0, n, block_size):
                    k_end = min(k + block_size, n)
                    c[i:i_end, j:j_end] += (
                        a64[i:i_end, k:k_end] @ b64[k:k_end, j:j_end]
                    )
    elapsed = time.perf_counter() - start
    return BlockedSerialResult(
        matrix=c,
        elapsed_seconds=elapsed,
        block_size=block_size,
    )
