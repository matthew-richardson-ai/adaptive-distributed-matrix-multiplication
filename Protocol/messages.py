from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


DTYPE = np.dtype("<f8")


def as_float64_c(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=DTYPE)


def matrix_to_bytes(array: np.ndarray) -> bytes:
    return as_float64_c(array).tobytes(order="C")


def matrix_from_bytes(data: bytes, shape: Iterable[int]) -> np.ndarray:
    shape_tuple = tuple(int(x) for x in shape)
    expected = int(np.prod(shape_tuple, dtype=np.int64)) * DTYPE.itemsize
    if len(data) != expected:
        raise ValueError(
            f"Payload length {len(data)} does not match shape {shape_tuple} "
            f"({expected} bytes expected)."
        )
    return np.frombuffer(data, dtype=DTYPE).reshape(shape_tuple).copy()


def pack_two_matrices(a: np.ndarray, b: np.ndarray) -> Tuple[bytes, dict]:
    a = as_float64_c(a)
    b = as_float64_c(b)
    a_bytes = a.tobytes(order="C")
    b_bytes = b.tobytes(order="C")
    metadata = {
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
        "a_nbytes": len(a_bytes),
        "b_nbytes": len(b_bytes),
        "dtype": "float64",
    }
    return a_bytes + b_bytes, metadata


def unpack_two_matrices(payload: bytes, metadata: dict) -> Tuple[np.ndarray, np.ndarray]:
    a_nbytes = int(metadata["a_nbytes"])
    b_nbytes = int(metadata["b_nbytes"])
    if a_nbytes < 0 or b_nbytes < 0 or a_nbytes + b_nbytes != len(payload):
        raise ValueError("Invalid two-matrix payload lengths.")

    a = matrix_from_bytes(payload[:a_nbytes], metadata["a_shape"])
    b = matrix_from_bytes(payload[a_nbytes:], metadata["b_shape"])
    return a, b
