"""判定者モデルが「比較の公平性」を保っているかを確かめる。

記事の主張は「平均エラー率が同じでも」という前提に立つので、
相関を入れても平均が変わらないことをテストで保証しておく。
"""

import numpy as np
import pytest

from fallible.combine import double_check, escape_rate_same_rate
from fallible.judges import JudgeSpec, correlated_with, simulate_daily_counts, simulate_errors
from fallible.tasks import beta_params

N_ITEMS, N_DAYS, P = 1000, 3000, 0.05


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("rho", [0.0, 0.005, 0.02, 0.05])
def test_mean_error_rate_is_preserved(rng, rho):
    """rho を変えても平均エラー率は p のまま。"""
    errors, _ = simulate_errors(rng, JudgeSpec("x", P, rho), N_DAYS, N_ITEMS)
    assert errors.mean() == pytest.approx(P, abs=0.003)


@pytest.mark.parametrize("rho", [0.005, 0.02])
def test_daily_variance_matches_beta_binomial(rng, rho):
    """日次誤り件数の分散がベータ二項分布の理論値 N p(1-p)[1+(N-1)rho] に近い。"""
    counts = simulate_daily_counts(rng, JudgeSpec("x", P, rho), 20000, N_ITEMS)
    theory = N_ITEMS * P * (1 - P) * (1 + (N_ITEMS - 1) * rho)
    assert counts.var(ddof=1) == pytest.approx(theory, rel=0.08)


def test_beta_params_mean():
    a, b = beta_params(P, 0.02)
    assert a / (a + b) == pytest.approx(P)


@pytest.mark.parametrize("phi", [0.0, 0.3, 0.8, 1.0])
def test_correlated_judge_keeps_rate_and_hits_phi(rng, phi):
    """2人目は平均エラー率を保ち、誤りの相関が phi になる。"""
    a, rates = simulate_errors(rng, JudgeSpec("x", P), 1000, N_ITEMS)
    b = correlated_with(rng, a, rates, phi)
    assert b.mean() == pytest.approx(P, abs=0.003)
    measured = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    assert measured == pytest.approx(phi, abs=0.02)


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_escape_rate_matches_theory(rng, phi):
    """二重チェックのすり抜け率が理論値 phi*q + (1-phi)*q^2 と一致する。"""
    a, rates = simulate_errors(rng, JudgeSpec("x", P), 2000, N_ITEMS)
    b = correlated_with(rng, a, rates, phi)
    assert double_check(a, b).mean() == pytest.approx(escape_rate_same_rate(P, phi), rel=0.08)
