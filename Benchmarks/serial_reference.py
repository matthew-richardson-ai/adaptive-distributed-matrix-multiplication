from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


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
