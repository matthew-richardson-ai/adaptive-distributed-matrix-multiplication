from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeneratedPair:
    a: np.ndarray
    b: np.ndarray
    seed: int
    matrix_type: str


def generate_dense(size: int, seed: int) -> GeneratedPair:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((size, size), dtype=np.float64)
    b = rng.standard_normal((size, size), dtype=np.float64)
    return GeneratedPair(a=a, b=b, seed=seed, matrix_type="dense")


def generate_block_sparse(
    size: int,
    seed: int,
    block_size: int,
    zero_block_fraction: float = 0.5,
) -> GeneratedPair:
    if not 0.0 <= zero_block_fraction <= 1.0:
        raise ValueError("zero_block_fraction must be between 0 and 1.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((size, size), dtype=np.float64)
    b = rng.standard_normal((size, size), dtype=np.float64)

    def zero_blocks(matrix: np.ndarray) -> None:
        blocks = []
        for r in range(0, size, block_size):
            for c in range(0, size, block_size):
                blocks.append((r, c))
        rng.shuffle(blocks)
        count = int(round(len(blocks) * zero_block_fraction))
        for r, c in blocks[:count]:
            matrix[r : min(r + block_size, size), c : min(c + block_size, size)] = 0.0

    zero_blocks(a)
    zero_blocks(b)
    return GeneratedPair(a=a, b=b, seed=seed, matrix_type="block_sparse")


def generate_low_rank(size: int, seed: int, rank: int) -> GeneratedPair:
    if rank <= 0 or rank > size:
        raise ValueError("rank must be in [1, size].")
    rng = np.random.default_rng(seed)

    a_left = rng.standard_normal((size, rank), dtype=np.float64)
    a_right = rng.standard_normal((size, rank), dtype=np.float64)
    b_left = rng.standard_normal((size, rank), dtype=np.float64)
    b_right = rng.standard_normal((size, rank), dtype=np.float64)

    a = a_left @ a_right.T
    b = b_left @ b_right.T
    return GeneratedPair(a=a, b=b, seed=seed, matrix_type="low_rank")


def generate_pair(
    matrix_type: str,
    size: int,
    seed: int,
    block_size: int = 128,
    zero_block_fraction: float = 0.5,
    rank: int = 8,
) -> GeneratedPair:
    if size <= 0:
        raise ValueError("size must be positive.")

    if matrix_type == "dense":
        return generate_dense(size, seed)
    if matrix_type == "block_sparse":
        return generate_block_sparse(
            size=size,
            seed=seed,
            block_size=block_size,
            zero_block_fraction=zero_block_fraction,
        )
    if matrix_type == "low_rank":
        return generate_low_rank(size=size, seed=seed, rank=rank)

    raise ValueError(f"Unsupported matrix type: {matrix_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic benchmark matrices.")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--matrix-type",
        choices=["dense", "block_sparse", "low_rank"],
        default="dense",
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--zero-block-fraction", type=float, default=0.5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output", default=None, help="Optional .npz output path.")
    args = parser.parse_args()

    generated = generate_pair(
        matrix_type=args.matrix_type,
        size=args.size,
        seed=args.seed,
        block_size=args.block_size,
        zero_block_fraction=args.zero_block_fraction,
        rank=args.rank,
    )

    print(
        f"Generated {generated.matrix_type} matrices "
        f"A={generated.a.shape}, B={generated.b.shape}, seed={generated.seed}"
    )

    if args.output:
        np.savez_compressed(
            args.output,
            a=generated.a,
            b=generated.b,
            seed=np.array([generated.seed]),
            matrix_type=np.array([generated.matrix_type]),
        )
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
