"""判定者モデル。

どの判定者も「各件を誤ったか（True/False）」の行列を返す。形は (日数, 1日の件数)。

- 人間型（独立誤り）：rho=0。各件が確率 p で独立に誤る。
- AI型（相関誤り）　：rho>0。日ごとにエラー率がゆらぎ、誤りが特定の日にまとまる。

2人目の判定者は「1人目と同じ件で誤りやすさ」を phi で指定して作る（correlated_with）。
phi は2人の誤り指標のファイ係数（相関係数）に一致する（周辺エラー率が等しい場合）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tasks import daily_error_rates


@dataclass(frozen=True)
class JudgeSpec:
    """判定者の性質。p は平均エラー率、rho は日をまたぐ誤りのまとまりやすさ。"""

    name: str
    p: float
    rho: float = 0.0


def simulate_errors(
    rng: np.random.Generator, spec: JudgeSpec, n_days: int, n_items: int
) -> tuple[np.ndarray, np.ndarray]:
    """1人の判定者の誤り行列 (n_days, n_items) と、日ごとのエラー率を返す。"""
    rates = daily_error_rates(rng, spec.p, spec.rho, n_days)
    errors = rng.random((n_days, n_items)) < rates[:, None]
    return errors, rates


def correlated_with(
    rng: np.random.Generator,
    first_errors: np.ndarray,
    rates: np.ndarray,
    phi: float,
) -> np.ndarray:
    """1人目の誤り行列と相関 phi を持つ2人目の誤り行列を作る。

    各件について、確率 phi で「1人目と同じ結果（同じ所で誤る／正解する）」を写し、
    残りは同じ日次エラー率で独立に引き直す。
    こうすると2人目の平均エラー率は1人目と同じまま、同じ日の中での誤り同士の相関がちょうど phi になる。
    （2人は日次エラー率 rates も共有するので、全期間で測った相関は rho の分だけさらに大きくなる。）
    同じモデルのAIを2つ並べた場合（同じ弱点を共有）を表すのに使う。
    """
    if not 0.0 <= phi <= 1.0:
        raise ValueError("phi must be in [0, 1]")
    copy_mask = rng.random(first_errors.shape) < phi
    fresh = rng.random(first_errors.shape) < rates[:, None]
    return np.where(copy_mask, first_errors, fresh)


def simulate_daily_counts(
    rng: np.random.Generator, spec: JudgeSpec, n_days: int, n_items: int
) -> np.ndarray:
    """日ごとの誤り件数だけが要るときの高速版。simulate_errors の行和と同じ分布になる。"""
    rates = daily_error_rates(rng, spec.p, spec.rho, n_days)
    return rng.binomial(n_items, rates)


def daily_counts(errors: np.ndarray) -> np.ndarray:
    """誤り行列から日ごとの誤り件数を返す。"""
    return errors.sum(axis=1)
