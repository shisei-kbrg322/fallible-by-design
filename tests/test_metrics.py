import numpy as np
import pytest

from fallible.combine import escape_rate_same_rate, majority_vote
from fallible.metrics import error_phi, exceed_prob, summarize_daily


def test_summarize_daily():
    s = summarize_daily(np.arange(1, 101), n_items=1000)
    assert s["mean_count"] == pytest.approx(50.5)
    assert s["mean_rate"] == pytest.approx(0.0505)
    assert s["max_count"] == 100


def test_exceed_prob():
    assert exceed_prob(np.array([1, 2, 3, 4]), 2) == pytest.approx(0.5)


def test_error_phi_identical_is_one():
    x = np.array([[True, False, True, False]])
    assert error_phi(x, x) == pytest.approx(1.0)


def test_escape_rate_edges():
    assert escape_rate_same_rate(0.05, 0.0) == pytest.approx(0.0025)
    assert escape_rate_same_rate(0.05, 1.0) == pytest.approx(0.05)


def test_majority_vote():
    a = np.array([True, True, False])
    b = np.array([True, False, False])
    c = np.array([False, True, True])
    assert majority_vote(a, b, c).tolist() == [True, True, False]
    with pytest.raises(ValueError):
        majority_vote(a, b)
