from __future__ import annotations

import numpy as np


def dense_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Worker operands must be 2-D matrices.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible block shapes: {a.shape} and {b.shape}.")
    return np.ascontiguousarray(
        np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)
    )


def left_factored_multiply(x: np.ndarray, y: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x @ (y.T @ b))


def right_factored_multiply(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((a @ x) @ y.T)


def both_factored_multiply(
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
) -> np.ndarray:
    return np.ascontiguousarray(x_a @ (y_a.T @ x_b) @ y_b.T)
