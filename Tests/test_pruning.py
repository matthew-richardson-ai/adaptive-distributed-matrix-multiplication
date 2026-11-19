import numpy as np

from Ingress.adaptive import frobenius_product_bound, should_prune


def test_exact_zero_left_block_is_pruned():
    a = np.zeros((4, 4), dtype=np.float64)
    b = np.ones((4, 4), dtype=np.float64)
    decision = should_prune(a, b, epsilon=0.0)
    assert decision.prune
    assert decision.bound == 0.0


def test_dense_nonzero_block_is_not_pruned_at_zero_tolerance():
    a = np.eye(4, dtype=np.float64)
    b = np.eye(4, dtype=np.float64)
    decision = should_prune(a, b, epsilon=0.0)
    assert not decision.prune
    assert decision.bound > 0.0


def test_positive_epsilon_can_prune_small_blocks():
    a = np.full((2, 2), 1e-8, dtype=np.float64)
    b = np.full((2, 2), 1e-8, dtype=np.float64)
    bound = frobenius_product_bound(a, b)
    decision = should_prune(a, b, epsilon=bound * 1.01)
    assert decision.prune


def test_negative_epsilon_rejected():
    a = np.eye(2)
    b = np.eye(2)
    try:
        should_prune(a, b, epsilon=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative epsilon should raise ValueError.")
