"""評価指標。平均だけでなく「悪い日」の大きさを見る。"""

from __future__ import annotations

import numpy as np


def summarize_daily(counts: np.ndarray, n_items: int) -> dict[str, float]:
    """日次の誤り件数から、平均・ばらつき・悪い日の指標をまとめる。"""
    counts = np.asarray(counts)
    return {
        "mean_count": float(counts.mean()),
        "mean_rate": float(counts.mean() / n_items),
        "std_count": float(counts.std(ddof=1)),
        "p50_count": float(np.percentile(counts, 50)),
        "p99_count": float(np.percentile(counts, 99)),
        "max_count": float(counts.max()),
    }


def exceed_prob(counts: np.ndarray, threshold: float) -> float:
    """日次の誤り件数が threshold を超える日の割合。"""
    return float((np.asarray(counts) > threshold).mean())


def error_phi(errors_a: np.ndarray, errors_b: np.ndarray) -> float:
    """2人の誤り指標（0/1）のファイ係数＝相関係数。"""
    a = errors_a.ravel().astype(float)
    b = errors_b.ravel().astype(float)
    return float(np.corrcoef(a, b)[0, 1])
