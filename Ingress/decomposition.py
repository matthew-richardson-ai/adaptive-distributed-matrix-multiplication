from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple
import uuid

import numpy as np

from Benchmarks.serial_reference import validate_multiplication_shapes


@dataclass(frozen=True)
class BlockTask:
    task_id: str
    job_id: str
    i: int
    j: int
    k: int
    a_block: np.ndarray
    b_block: np.ndarray
    row_start: int
    col_start: int


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("block_size must be positive.")
    return ((value + multiple - 1) // multiple) * multiple


def pad_for_blocking(
    a: np.ndarray,
    b: np.ndarray,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    validate_multiplication_shapes(a, b)

    m, n = a.shape
    _, p = b.shape

    padded_m = _ceil_to_multiple(m, block_size)
    padded_n = _ceil_to_multiple(n, block_size)
    padded_p = _ceil_to_multiple(p, block_size)

    a_pad = np.zeros((padded_m, padded_n), dtype=np.float64)
    b_pad = np.zeros((padded_n, padded_p), dtype=np.float64)
    a_pad[:m, :n] = np.asarray(a, dtype=np.float64)
    b_pad[:n, :p] = np.asarray(b, dtype=np.float64)

    return a_pad, b_pad, (m, p)


def iter_block_tasks(
    a: np.ndarray,
    b: np.ndarray,
    block_size: int,
    job_id: str,
) -> Iterator[BlockTask]:
    a_pad, b_pad, _ = pad_for_blocking(a, b, block_size)

    block_rows = a_pad.shape[0] // block_size
    block_inner = a_pad.shape[1] // block_size
    block_cols = b_pad.shape[1] // block_size

    for i in range(block_rows):
        for j in range(block_cols):
            for k in range(block_inner):
                rs = i * block_size
                re = rs + block_size
                ks = k * block_size
                ke = ks + block_size
                cs = j * block_size
                ce = cs + block_size

                yield BlockTask(
                    task_id=str(uuid.uuid4()),
                    job_id=job_id,
                    i=i,
                    j=j,
                    k=k,
                    a_block=np.ascontiguousarray(a_pad[rs:re, ks:ke]),
                    b_block=np.ascontiguousarray(b_pad[ks:ke, cs:ce]),
                    row_start=rs,
                    col_start=cs,
                )


def count_block_tasks(a: np.ndarray, b: np.ndarray, block_size: int) -> int:
    a_pad, b_pad, _ = pad_for_blocking(a, b, block_size)
    return (
        (a_pad.shape[0] // block_size)
        * (a_pad.shape[1] // block_size)
        * (b_pad.shape[1] // block_size)
    )
