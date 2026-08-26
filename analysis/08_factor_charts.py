# -*- coding: utf-8 -*-
"""中性化条件因子(GWS_full)可视化: 多空/多头超额净值, 分年度收益,
IC/RankIC 序列与汇总指标. 图表输出至 figures/v1_monthly_factor/.

口径说明:
  - 分组: 每月末按 GWS_full 十分位, 收益为次月; 净值曲线按收益所在月标注
  - 多空 = G10 - G1 (等权, 月调仓, 无成本); 多头超额 = G10 - 全样本等权
  - 净值 = ∏(1+月收益); IC 滚动线为 12 个月均值
运行: 需 pyarrow + matplotlib (见 README)
"""
import _path  # noqa: F401
import sys

try:
    import pyarrow  # noqa: F401
except ImportError:
    sys.exit("需要 pyarrow: pip install pyarrow")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from greenwash import config
from greenwash.factor import ic_series
from greenwash.pipeline import build_signal_panel

SIG = "GWS_full"
CHART_DIR = config.CHART_DIR

# dataviz 调色板 (light mode)
C = dict(blue="#2a78d6", orange="#eb6834", red="#e34948", aqua="#1baf7a",
         ink="#0b0b0b", ink2="#52514e", muted="#898781",
         grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb")

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "savefig.facecolor": C["surface"], "savefig.dpi": 150,
    "axes.edgecolor": C["axis"], "axes.linewidth": 0.8,
    "xtick.color": C["muted"], "ytick.color": C["muted"],
    "axes.labelcolor": C["ink2"], "text.color": C["ink"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def style_ax(ax, pct=False):
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    if pct:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{round(v*100, 2):g}%"))


def month_axis(ax, idx):
    years = sorted({m[:4] for m in idx})
    ticks = [next(i for i, m in enumerate(idx) if m.startswith(y))
             for y in years]
    ax.set_xticks(ticks)
    ax.set_xticklabels(years)


if __name__ == "__main__":
    print(">> 构建/加载月频信号面板 ...", flush=True)
    d = build_signal_panel()

    # ---- 分组收益序列 (收益记在次月) ----
    dd = d.dropna(subset=[SIG, "fwd_ret"]).copy()
    dd["grp"] = dd.groupby("month")[SIG].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    gm = dd.pivot_table(index="month", columns="grp", values="fwd_ret")
    uni = dd.groupby("month")["fwd_ret"].mean()
    ret_month = (pd.PeriodIndex(gm.index, freq="M") + 1).astype(str)

    ls = (gm[10] - gm[1]).set_axis(ret_month).dropna()
    lexc = (gm[10] - uni).set_axis(ret_month).dropna()
    nav_ls, nav_le = (1 + ls).cumprod(), (1 + lexc).cumprod()
    mdd = float((nav_ls / nav_ls.cummax() - 1).min())

    ics = ic_series(d, SIG, "fwd_ret", time="month", min_n=100)
    ics.index = (pd.PeriodIndex(ics.index, freq="M") + 1).astype(str)

    def ann(s):
        t, _ = stats.ttest_1samp(s, 0)
        return s.mean() * 12, s.std() * np.sqrt(12), t

    ls_a, ls_v, ls_t = ann(ls)
    metrics = pd.Series({
        "IC": ics["IC"].mean(), "ICIR": ics["IC"].mean() / ics["IC"].std(),
        "IC_t": ics["IC"].mean() / ics["IC"].std() * np.sqrt(len(ics)),
        "RankIC": ics["RankIC"].mean(),
        "RankICIR": ics["RankIC"].mean() / ics["RankIC"].std(),
        "RankIC_t": ics["RankIC"].mean() / ics["RankIC"].std() * np.sqrt(len(ics)),
        "多空年化": ls_a, "多空年化波动": ls_v, "多空Sharpe": ls_a / ls_v,
        "多空t": ls_t, "月度胜率": (ls > 0).mean(), "最大回撤": mdd,
        "多头超额年化": lexc.mean() * 12, "月数": len(ls),
    })
    print(metrics.round(4).to_string())
    metrics.round(6).to_csv(config.OUT_DIR / "factor_gws_full_metrics.csv",
                            encoding="utf-8-sig")
    pd.DataFrame({"多空净值": nav_ls, "多头超额净值": nav_le}).to_csv(
        config.OUT_DIR / "factor_gws_full_nav.csv", encoding="utf-8-sig")

    # ---- 图1: 净值曲线 ----
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = range(len(nav_ls))
    ax.plot(x, nav_ls.values, lw=2, color=C["blue"], label="多空净值 (G10−G1)")
    ax.plot(x, nav_le.reindex(nav_ls.index).values, lw=2, color=C["orange"],
            label="多头超额净值 (G10−全样本)")
    for nav, c in [(nav_ls, C["blue"]), (nav_le, C["orange"])]:
        ax.annotate(f"{nav.iloc[-1]:.2f}", (len(nav_ls) - 1, nav.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=c, fontweight="bold", va="center")
    ax.axhline(1, color=C["axis"], lw=0.8)
    month_axis(ax, nav_ls.index)
    style_ax(ax)
    ax.set_title("中性化漂绿因子 GWS_full: 累计净值 (月调仓·等权·无成本)",
                 loc="left", fontsize=12, pad=12)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_nav_curves.png")

    # ---- 图2: 分年度收益 ----
    yr_ls = (1 + ls).groupby(ls.index.str[:4]).prod() - 1
    yr_le = (1 + lexc).groupby(lexc.index.str[:4]).prod() - 1
    years = yr_ls.index
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = np.arange(len(years))
    for off, ser, c, lab in [(-0.2, yr_ls, C["blue"], "多空 (G10−G1)"),
                             (0.2, yr_le, C["orange"], "多头超额")]:
        v = ser.reindex(years).values
        ax.bar(xs + off, v, 0.36, color=c, label=lab)
        for xi, vi in zip(xs + off, v):
            ax.annotate(f"{vi:+.1%}", (xi, vi),
                        xytext=(0, 4 if vi >= 0 else -12),
                        textcoords="offset points", ha="center",
                        fontsize=8, color=C["ink2"])
    ax.axhline(0, color=C["axis"], lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(years)
    ax.margins(y=0.18)
    style_ax(ax, pct=True)
    ax.set_title("分年度收益", loc="left", fontsize=12, pad=12)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_annual_returns.png")

    # ---- 图3: IC / RankIC 序列 ----
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    for ax, col in zip(axes, ["IC", "RankIC"]):
        s = ics[col]
        colors = [C["blue"] if v >= 0 else C["red"] for v in s]
        ax.bar(range(len(s)), s.values, 0.8, color=colors, alpha=0.55)
        roll = s.rolling(12).mean()
        ax.plot(range(len(s)), roll.values, lw=2, color=C["orange"],
                label="12个月滚动均值")
        ax.axhline(0, color=C["axis"], lw=0.8)
        ir = s.mean() / s.std()
        ax.set_title(f"月度{col}   均值 {s.mean():+.4f} · {col}IR {ir:.2f} · "
                     f"t={ir*np.sqrt(len(s)):.2f} · 胜率 {(s>0).mean():.0%}",
                     loc="left", fontsize=11, pad=8)
        style_ax(ax)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    month_axis(axes[1], ics.index)
    fig.suptitle("中性化漂绿因子 GWS_full: IC 序列", x=0.065, y=0.99,
                 ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(CHART_DIR / "03_ic_series.png")

    # ---- 图4: 十分组月均前瞻收益 ----
    fig, ax = plt.subplots(figsize=(7.5, 4))
    gv = gm.mean()
    ax.bar(gv.index.astype(int), gv.values, 0.72, color=C["blue"])
    ax.axhline(uni.mean(), color=C["muted"], lw=1.2, ls="--")
    ax.annotate(f"全样本均值 {uni.mean():.2%}", (0.6, uni.mean()),
                xytext=(0, 5), textcoords="offset points",
                fontsize=9, color=C["ink2"])
    ax.set_xticks(gv.index.astype(int))
    ax.set_xticklabels([f"G{i}" for i in gv.index.astype(int)])
    ax.set_xlabel("GWS_full 十分组 (G1 最低 → G10 最高)")
    style_ax(ax, pct=True)
    ax.set_title("十分组月均前瞻收益", loc="left", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "04_decile_returns.png")

    # ---- 图5: 指标卡 ----
    tiles = [("IC / ICIR", f"{metrics['IC']:+.4f} / {metrics['ICIR']:.2f}",
              f"t = {metrics['IC_t']:.2f}"),
             ("RankIC / RankICIR",
              f"{metrics['RankIC']:+.4f} / {metrics['RankICIR']:.2f}",
              f"t = {metrics['RankIC_t']:.2f}"),
             ("多空年化 / Sharpe",
              f"{metrics['多空年化']:+.1%} / {metrics['多空Sharpe']:.2f}",
              f"t = {metrics['多空t']:.2f}"),
             ("月度胜率 / 最大回撤",
              f"{metrics['月度胜率']:.0%} / {metrics['最大回撤']:.1%}",
              f"{int(metrics['月数'])} 个月 · 月调仓等权无成本")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 1.9))
    for ax, (k, v, sub) in zip(axes, tiles):
        ax.axis("off")
        ax.text(0, 0.85, k, fontsize=9.5, color=C["ink2"], va="top")
        ax.text(0, 0.48, v, fontsize=15, color=C["ink"], va="center",
                fontweight="bold")
        ax.text(0, 0.10, sub, fontsize=8.5, color=C["muted"], va="bottom")
    fig.suptitle("中性化漂绿因子 GWS_full · 2015-01 ~ 2024-12", x=0.02,
                 ha="left", fontsize=11, color=C["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(CHART_DIR / "05_metric_cards.png")

    print(f"\n图表已输出至 {CHART_DIR}/ (5 张 png)")
