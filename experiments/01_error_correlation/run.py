"""実験01：誤りの相関シミュレーション。

平均エラー率が同じでも、間違え方（誤りの相関）が違えば、
悪い日の大きさと二重チェックの効き方が変わることを確かめる。

使い方:
    docker compose run --rm sim                       # 既定の config.yaml で実行
    python experiments/01_error_correlation/run.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import FuncFormatter

from fallible.combine import double_check, escape_rate_same_rate
from fallible.judges import JudgeSpec, correlated_with, simulate_daily_counts, simulate_errors
from fallible.metrics import error_phi, exceed_prob, summarize_daily
from fallible.plotting import COLORS, DIVERGING, TEXT, TEXT_SUB, setup_style

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- 実験A
def experiment_a(cfg: dict, rng: np.random.Generator, out: Path) -> pd.DataFrame:
    """単独判定：rho を変えたときの日次誤り件数の分布。"""
    n, d, p = cfg["n_items"], cfg["n_days"], cfg["p"]
    ea = cfg["experiment_a"]

    counts_by_rho: dict[float, np.ndarray] = {}
    rows = []
    for rho in ea["rho_values"]:
        spec = JudgeSpec("人間型" if rho == 0 else "AI型", p=p, rho=rho)
        counts = simulate_daily_counts(rng, spec, d, n)
        counts_by_rho[rho] = counts
        s = summarize_daily(counts, n)
        rows.append({"type": spec.name, "rho": rho, **s})

    df = pd.DataFrame(rows)
    human_p99 = df.loc[df["rho"] == 0, "p99_count"].iloc[0]
    df["days_over_human_p99"] = [exceed_prob(counts_by_rho[r], human_p99) for r in df["rho"]]
    df.to_csv(out / "a_summary.csv", index=False, float_format="%.4f")

    # 図A：人間型と AI型（highlight_rho）の日次誤り件数の分布
    rho_h = ea["highlight_rho"]
    human, ai = counts_by_rho[0.0], counts_by_rho[rho_h]
    upper = int(np.percentile(ai, 99.7)) + 10
    bins = np.arange(0, upper + 2, 2)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    p99_marks: list[tuple[float, str]] = []
    for counts, key, label in [
        (human, "human", "人間型（独立に間違える）"),
        (ai, "ai", f"AI型（まとめて間違える, ρ={rho_h}）"),
    ]:
        hist, edges = np.histogram(counts, bins=bins)
        share = hist / len(counts) * 100
        ax.stairs(share, edges, color=COLORS[key], linewidth=2, label=label)
        ax.stairs(share, edges, color=COLORS[key], alpha=0.12, fill=True)
        p99 = np.percentile(counts, 99)
        ax.axvline(p99, color=COLORS[key], linestyle=(0, (4, 3)), linewidth=1.4)
        p99_marks.append((p99, key))
    ymax = ax.get_ylim()[1]
    for p99, key in p99_marks:
        ax.text(
            p99 + 2,
            ymax * (0.55 if key == "human" else 0.35),
            f"上位1%の悪い日\n{p99:.0f}件以上",
            color=COLORS[key],
            fontsize=9,
        )
    ax.axvline(n * p, color=TEXT, linewidth=1, alpha=0.5)
    ax.text(n * p + 2, ax.get_ylim()[1] * 0.97, f"平均 {n * p:.0f}件（共通）", color=TEXT, fontsize=9, va="top")
    ax.set_xlim(0, upper)
    ax.set_xlabel(f"1日の誤り件数（1日 {n:,}件の判定）")
    ax.set_ylabel("その件数になった日の割合（%）")
    ax.set_title("平均エラー率は同じ5%でも、「悪い日」の大きさがまったく違う")
    ax.legend(loc="upper right")
    fig.savefig(out / "a_daily_errors.png")
    plt.close(fig)
    return df


# ---------------------------------------------------------------- 実験B
def _pair_escape(
    rng: np.random.Generator, cfg: dict, kind: str, phi: float
) -> tuple[np.ndarray, float]:
    """二重チェックのすり抜け行列と、実測の誤りの相関を返す。"""
    n, d, p = cfg["n_items"], cfg["n_days"], cfg["p"]
    eb = cfg["experiment_b"]
    if kind == "human_human":
        a, rates = simulate_errors(rng, JudgeSpec("人間", p), d, n)
        b = correlated_with(rng, a, rates, phi)
    elif kind == "ai_ai":
        a, rates = simulate_errors(rng, JudgeSpec("AI", p, eb["ai_rho"]), d, n)
        b = correlated_with(rng, a, rates, phi)
    elif kind == "human_ai":
        a, _ = simulate_errors(rng, JudgeSpec("人間", p), d, n)
        b, _ = simulate_errors(rng, JudgeSpec("AI", p, eb["ai_rho"]), d, n)
        if phi > 0:
            raise ValueError("human_ai with phi>0 is not modeled in this experiment")
    else:
        raise ValueError(kind)
    return double_check(a, b), error_phi(a, b)


def experiment_b(cfg: dict, rng: np.random.Generator, out: Path) -> pd.DataFrame:
    """二重チェック：組み合わせごとのすり抜け率。"""
    n = cfg["n_items"]
    eb = cfg["experiment_b"]
    pairs = [
        ("human_ai", "人間 + AI", eb["human_ai_phi"]),
        ("human_human", "人間 + 人間", eb["human_pair_phi"]),
        ("ai_ai", "AI + AI（同じモデル）", eb["ai_pair_phi"]),
    ]
    rows = []
    for kind, label, phi in pairs:
        escaped, measured_phi = _pair_escape(rng, cfg, kind, phi)
        counts = escaped.sum(axis=1)
        rows.append(
            {
                "pair": kind,
                "label": label,
                "phi_within_day": phi,
                "phi_measured": measured_phi,
                "escape_rate": float(escaped.mean()),
                "escape_per_1000": float(escaped.mean() * 1000),
                "p99_daily_escape": float(np.percentile(counts, 99)),
            }
        )
        del escaped
    df = pd.DataFrame(rows)
    df.to_csv(out / "b_summary.csv", index=False, float_format="%.5f")

    # 感度：人間同士の相関を変えたときの「人間 + 人間」（理論値）と「人間 + AI」の比較
    p = cfg["p"]
    sens = pd.DataFrame(
        {
            "human_pair_phi": eb["sensitivity_human_pair_phi"],
            "human_human_escape_per_1000": [
                escape_rate_same_rate(p, f) * 1000 for f in eb["sensitivity_human_pair_phi"]
            ],
            "human_ai_escape_per_1000": p * p * 1000,
        }
    )
    sens.to_csv(out / "b_sensitivity.csv", index=False, float_format="%.3f")

    # 図B：1,000件あたりのすり抜け件数
    fig, ax = plt.subplots(figsize=(8, 3.2))
    color_of = {"human_ai": COLORS["human_ai"], "human_human": COLORS["human"], "ai_ai": COLORS["ai"]}
    y = np.arange(len(df))[::-1]
    ax.barh(y, df["escape_per_1000"], color=[color_of[k] for k in df["pair"]], height=0.56)
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.text(
            r["escape_per_1000"] + 0.6,
            yi,
            f"{r['escape_per_1000']:.1f}件",
            va="center",
            color=TEXT,
            fontsize=10.5,
            fontweight="bold",
        )
    ax.set_yticks(y, [f"{r['label']}\n（誤りの相関 φ={r['phi_within_day']:.1f}）" for _, r in df.iterrows()])
    ax.set_xlabel(f"1,000件あたりのすり抜け件数（各判定者の平均エラー率はすべて{p:.0%}）")
    ax.set_xlim(0, df["escape_per_1000"].max() * 1.18)
    ax.grid(axis="y", visible=False)
    ax.set_title("同じAIを2回通しても、二重チェックはほとんど効かない")
    fig.savefig(out / "b_double_check.png")
    plt.close(fig)
    return df


# ---------------------------------------------------------------- 実験C
def experiment_c(cfg: dict, rng: np.random.Generator, out: Path) -> pd.DataFrame:
    """感度分析：AI単体のエラー率 × AI同士の誤りの相関 → 二重チェック後のすり抜け率。"""
    p = cfg["p"]
    ec = cfg["experiment_c"]
    qs = np.linspace(ec["q_min"], ec["q_max"], ec["q_steps"])
    phis = np.linspace(0.0, 1.0, ec["phi_steps"])
    Q, PHI = np.meshgrid(qs, phis)
    escape = PHI * Q + (1 - PHI) * Q * Q
    baseline = escape_rate_same_rate(p, ec["baseline_human_pair_phi"])
    ratio = escape / baseline

    # 理論値をシミュレーションで確かめる
    vrows = []
    for q in np.linspace(ec["q_min"] * 2, ec["q_max"], ec["validate_points"]):
        for phi in (0.0, 0.5, 0.9):
            a, rates = simulate_errors(rng, JudgeSpec("AI", q), ec["validate_days"], cfg["n_items"])
            b = correlated_with(rng, a, rates, phi)
            vrows.append(
                {
                    "q": q,
                    "phi": phi,
                    "escape_theory": escape_rate_same_rate(q, phi),
                    "escape_simulated": float(double_check(a, b).mean()),
                }
            )
    vdf = pd.DataFrame(vrows)
    vdf.to_csv(out / "c_validation.csv", index=False, float_format="%.6f")

    # 図C：基準（人間2人）に対する倍率。対数の発散スケールで、1倍＝灰
    fig, ax = plt.subplots(figsize=(8, 5.2))
    lim = 1.0  # 0.1倍〜10倍で色を飽和させ、境界付近の差を見やすくする
    mesh = ax.pcolormesh(Q * 100, PHI, np.log10(ratio), cmap=DIVERGING, vmin=-lim, vmax=lim, shading="auto")
    ax.contour(Q * 100, PHI, ratio, levels=[1.0], colors=[TEXT], linewidths=1.8)
    # 境界線のラベル：曲線上の点 q=1.6% に横書きで添える
    q_lab = 0.016
    phi_lab = (baseline - q_lab**2) / (q_lab - q_lab**2)
    ax.annotate(
        "人間2人と同等の境界",
        xy=(q_lab * 100, phi_lab),
        xytext=(22, 10),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": TEXT, "lw": 1},
        fontsize=9.5,
        color=TEXT,
        fontweight="bold",
    )
    ax.text(0.8, 0.06, "人間2人より\nすり抜けが少ない", color="#104281", fontsize=9.5, fontweight="bold")
    ax.text(7.2, 0.45, "人間2人より\nすり抜けが多い", color="white", fontsize=9.5, fontweight="bold")
    ax.axvline(p * 100, color=TEXT, linestyle=(0, (4, 3)), linewidth=1, alpha=0.6)
    ax.text(p * 100 + 0.15, 0.965, f"人間のエラー率 {p:.0%}", color=TEXT, fontsize=9, va="top")
    for m in ec["markers"]:
        r = escape_rate_same_rate(m["q"], m["phi"]) / baseline
        ax.plot(m["q"] * 100, m["phi"], "o", ms=8, mfc="white", mec=TEXT, mew=1.6)
        ax.annotate(
            f"{m['label']}\n人間2人の{r:.1f}倍すり抜け",
            xy=(m["q"] * 100, m["phi"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color=TEXT,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.85},
        )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, extend="both")
    ticks = np.log10([0.1, 0.3, 1, 3, 10])
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(["0.1倍", "0.3倍", "1倍", "3倍", "10倍"])
    cbar.set_label("すり抜け件数（人間2人の二重チェック＝1倍）", color=TEXT_SUB)
    cbar.outline.set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}%"))
    ax.set_xlabel("AI単体のエラー率")
    ax.set_ylabel("2つのAIの誤りの相関 φ（同じモデルほど高い）")
    ax.grid(False)
    ax.set_title("AI単体が人間より優秀でも、相関が高いと二重チェック後に逆転する")
    fig.savefig(out / "c_heatmap.png")
    plt.close(fig)

    # 境界：各 q で人間2人と同等になる phi
    boundary = []
    for q in [0.01, 0.02, 0.03, 0.04, 0.05]:
        # phi*q + (1-phi)q^2 = baseline を phi について解く
        phi_star = (baseline - q * q) / (q - q * q)
        boundary.append({"q": q, "phi_break_even": float(np.clip(phi_star, 0, 1))})
    bdf = pd.DataFrame(boundary)
    bdf.to_csv(out / "c_break_even.csv", index=False, float_format="%.4f")
    return bdf


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()

    # 実験ごとに独立した乱数列を使い、どれかを変えても他の結果が動かないようにする
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(3)
    rngs = [np.random.default_rng(s) for s in seeds]

    pd.set_option("display.width", 140)
    print("== 実験A：単独判定 ==")
    print(experiment_a(cfg, rngs[0], args.out).to_string(index=False))
    print("\n== 実験B：二重チェック ==")
    print(experiment_b(cfg, rngs[1], args.out).to_string(index=False))
    print("\n== 実験C：人間2人と同等になる相関（これを超えるとAI×2が負ける）==")
    print(experiment_c(cfg, rngs[2], args.out).to_string(index=False))
    print(f"\n結果と図を {args.out} に出力しました。")


if __name__ == "__main__":
    main()
