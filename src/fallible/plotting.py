"""図の体裁をそろえる。色は役割で固定する（人間＝青、AI＝橙、人間＋AI＝緑）。"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

COLORS = {
    "human": "#2a78d6",
    "ai": "#eb6834",
    "human_ai": "#1baf7a",
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SUB = "#52514e"
GRID = "#e4e3df"

# 発散カラーマップ：青（人間2人より良い）― 灰（同等）― 赤（悪い）
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#86b6ef", "#f0efec", "#ec8a89", "#a52a2a"]
)

_JP_FONTS = ["Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "Hiragino Sans", "Yu Gothic"]


def setup_style() -> None:
    """日本語フォントと、控えめな軸・グリッドを設定する。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = [f for f in _JP_FONTS if f in available] or ["DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.family": family + ["DejaVu Sans"],
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SUB,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SUB,
            "ytick.color": TEXT_SUB,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
