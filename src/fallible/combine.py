"""判定の組み合わせ方。

二重チェックの前提：2人の判定が食い違えば、その件は見直されて正しく処理される。
したがって最終的な誤りとして残る（すり抜ける）のは「2人とも誤った件」だけ。
二値の判定なので、2人とも誤ったときは同じ誤答で一致し、食い違いとして検出されない。
"""

from __future__ import annotations

import numpy as np


def double_check(errors_a: np.ndarray, errors_b: np.ndarray) -> np.ndarray:
    """2人とも誤った件（すり抜け）を True とする行列を返す。"""
    return errors_a & errors_b


def majority_vote(*errors: np.ndarray) -> np.ndarray:
    """奇数人の多数決で、過半数が誤った件を True とする行列を返す。"""
    if len(errors) % 2 == 0:
        raise ValueError("majority_vote needs an odd number of judges")
    stacked = np.stack(errors, axis=0)
    return stacked.sum(axis=0) > len(errors) // 2


def escape_rate_same_rate(q: float, phi: float) -> float:
    """平均エラー率 q が等しく、誤りの相関が phi の2者による二重チェックのすり抜け率（理論値）。

    P(両方誤る) = phi * q + (1 - phi) * q^2
    phi=0（独立）なら q^2、phi=1（完全に同じ間違え方）なら q となり、二重チェックの効果が消える。
    """
    return phi * q + (1.0 - phi) * q * q
