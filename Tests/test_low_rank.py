import numpy as np

from Ingress.adaptive import factor_payload_is_smaller, truncated_svd_factors
from Worker.kernels import (
    both_factored_multiply,
    left_factored_multiply,
    right_factored_multiply,
)


def test_rank_one_factorization_is_detected():
    x = np.arange(1, 9, dtype=np.float64).reshape(8, 1)
    y = np.arange(2, 10, dtype=np.float64).reshape(8, 1)
    block = x @ y.T

    factors = truncated_svd_factors(block, tolerance=1e-10, max_rank=4)
    assert factors is not None
    assert factors.rank == 1
    assert factors.relative_error <= 1e-10
    assert factor_payload_is_smaller(block, factors)
    assert np.allclose(factors.x @ factors.y.T, block)


def test_factored_worker_paths_match_dense_product():
    rng = np.random.default_rng(42)
    xa = rng.standard_normal((6, 2))
    ya = rng.standard_normal((6, 2))
    xb = rng.standard_normal((6, 2))
    yb = rng.standard_normal((6, 2))

    a = xa @ ya.T
    b = xb @ yb.T
    expected = a @ b

    assert np.allclose(left_factored_multiply(xa, ya, b), expected)
    assert np.allclose(right_factored_multiply(a, xb, yb), expected)
    assert np.allclose(both_factored_multiply(xa, ya, xb, yb), expected)
