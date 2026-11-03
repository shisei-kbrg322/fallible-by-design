"""タスク（1日分の判定対象）側の生成。

日ごとに入力の構成がゆらぐと、同じ判定者でも「その日のエラー率」が上下する。
このゆらぎの大きさを級内相関 rho で表す。
"""

from __future__ import annotations

import numpy as np


def beta_params(p: float, rho: float) -> tuple[float, float]:
    """平均 p・級内相関 rho のベータ分布パラメータ (a, b) を返す。

    1日 N 件の誤り件数は、このベータ分布で日次エラー率を引いたうえでの
    二項分布（＝ベータ二項分布）になり、分散は N p (1-p) [1 + (N-1) rho] となる。
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must be in (0, 1); use rho=0 via daily_error_rates")
    k = (1.0 - rho) / rho
    return p * k, (1.0 - p) * k


def daily_error_rates(
    rng: np.random.Generator, p: float, rho: float, n_days: int
) -> np.ndarray:
    """日ごとのエラー率を n_days 個返す。rho=0 なら全日 p で一定（独立誤り）。"""
    if rho == 0.0:
        return np.full(n_days, p)
    a, b = beta_params(p, rho)
    return rng.beta(a, b, size=n_days)
