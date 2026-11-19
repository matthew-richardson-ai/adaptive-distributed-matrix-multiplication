import numpy as np
import pytest

from Benchmarks.serial_reference import (
    multiply_blocked_serial,
    multiply_serial,
    validate_multiplication_shapes,
    validate_result,
)


def test_serial_reference_matches_numpy():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    result = multiply_serial(a, b)
    assert np.allclose(result.matrix, a @ b)
    assert result.elapsed_seconds >= 0.0


def test_invalid_shapes_raise():
    a = np.ones((2, 3))
    b = np.ones((2, 2))
    with pytest.raises(ValueError):
        validate_multiplication_shapes(a, b)


def test_validate_result():
    a = np.eye(3)
    b = a + 1e-13
    assert validate_result(a, b)


def test_blocked_serial_matches_numpy():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((7, 5))
    b = rng.standard_normal((5, 6))
    result = multiply_blocked_serial(a, b, block_size=3)
    assert result.block_size == 3
    assert result.elapsed_seconds >= 0.0
    assert np.allclose(result.matrix, a @ b)


def test_blocked_serial_rejects_invalid_block_size():
    a = np.eye(2)
    with pytest.raises(ValueError):
        multiply_blocked_serial(a, a, block_size=0)
