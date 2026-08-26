# -*- coding: utf-8 -*-
"""全市场增强应用: 低披露尾部负面清单 (GWS_full 旗标型用法).

组合设定 (等权, 月调仓, 无成本):
  基准   = 信号覆盖域等权 (月均 ~970 只)
  增强   = 基准 剔除 GWS_full 尾部 10% 旗标名单
  两个清单版本: 月度刷新 (每月重算旗标) / 年度冻结 (每年 1 月定名单持有 12 个月,
  实盘负面清单的真实形态, 换手最低)
输出: 超额净值/分年度超额/清单换手/完整指标(含胜率/盈亏比/盈利因子), 图表 06.
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
from greenwash.pipeline import build_signal_panel

CHART_DIR = config.CHART_DIR
C = dict(blue="#2a78d6", orange="#eb6834", ink="#0b0b0b", ink2="#52514e",
         muted="#898781", grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb")
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "savefig.facecolor": C["surface"], "savefig.dpi": 150,
    "axes.edgecolor": C["axis"], "xtick.color": C["muted"],
    "ytick.color": C["muted"], "axes.labelcolor": C["ink2"],
    "text.color": C["ink"], "axes.grid": True, "grid.color": C["grid"],
    "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10,
})


def metrics(exc, name):
    """月度超额序列 → 指标 (含单笔口径: 触发=月)."""
    t, p = stats.ttest_1samp(exc, 0)
    nav = (1 + exc).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())
    return pd.Series({
        "年化超额": exc.mean() * 12, "跟踪误差": exc.std() * np.sqrt(12),
        "信息比率": exc.mean() / exc.std() * np.sqrt(12), "t": t, "p": p,
        "触发月数": len(exc), "月胜率": (exc > 0).mean(),
        "单月均超额": exc.mean(),
        "盈亏比": exc[exc > 0].mean() / abs(exc[exc < 0].mean()),
        "盈利因子": exc[exc > 0].sum() / abs(exc[exc < 0].sum()),
        "最大相对回撤": mdd,
    }, name=name)


if __name__ == "__main__":
    print(">> 构建信号面板 ...", flush=True)
    d = build_signal_panel(verbose=False)
    dd = d.dropna(subset=["GWS_full", "fwd_ret"]).copy()
    dd["pct"] = dd.groupby("month")["GWS_full"].rank(pct=True)
    dd["flag_m"] = dd["pct"] <= 0.10           # 月度刷新旗标

    # 年度冻结: 每年首月的旗标名单持有全年
    first = dd.groupby(dd["month"].str[:4])["month"].transform("min")
    jan_list = dd[(dd["month"] == first) & dd["flag_m"]]
    frozen = set(map(tuple, jan_list[["code"]].assign(
        y=jan_list["month"].str[:4])[["y", "code"]].values))
    dd["flag_f"] = [(m[:4], c) in frozen for m, c in zip(dd["month"], dd["code"])]

    base = dd.groupby("month")["fwd_ret"].mean()
    enh_m = dd[~dd["flag_m"]].groupby("month")["fwd_ret"].mean()
    enh_f = dd[~dd["flag_f"]].groupby("month")["fwd_ret"].mean()
    exc_m, exc_f = (enh_m - base).dropna(), (enh_f - base).dropna()
    # 收益记在次月
    for s in (exc_m, exc_f):
        s.index = (pd.PeriodIndex(s.index, freq="M") + 1).astype(str)

    # 清单换手 (相邻月旗标名单变动比例)
    lists = dd[dd["flag_m"]].groupby("month")["code"].apply(set)
    turn_m = np.mean([len(a ^ b) / max(len(a | b), 1)
                      for a, b in zip(lists[:-1], lists[1:])])
    lists_f = dd[dd["flag_f"]].groupby("month")["code"].apply(set)
    turn_f = np.mean([len(a ^ b) / max(len(a | b), 1)
                      for a, b in zip(lists_f[:-1], lists_f[1:])])

    tab = pd.concat([metrics(exc_m, "月度刷新清单"),
                     metrics(exc_f, "年度冻结清单")], axis=1)
    tab.loc["月均剔除只数"] = [dd["flag_m"].groupby(dd["month"]).sum().mean(),
                          dd["flag_f"].groupby(dd["month"]).sum().mean()]
    tab.loc["清单月均换手"] = [turn_m, turn_f]
    print("\n== 全市场增强: 剔除低披露尾部10% 的超额表现 ==")
    print(tab.round(4).to_string())
    tab.round(6).to_csv(config.OUT_DIR / "enhance_fullmarket.csv",
                        encoding="utf-8-sig")

    yr = pd.DataFrame({"月度刷新": (1 + exc_m).groupby(exc_m.index.str[:4]).prod() - 1,
                       "年度冻结": (1 + exc_f).groupby(exc_f.index.str[:4]).prod() - 1})
    print("\n== 分年度超额 ==\n", (yr * 100).round(2).to_string())
    yr.round(6).to_csv(config.OUT_DIR / "enhance_fullmarket_yearly.csv",
                       encoding="utf-8-sig")

    # ---- 图 6: 超额净值 + 分年度超额 ----
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.6), height_ratios=[3, 2])
    nav_m, nav_f = (1 + exc_m).cumprod(), (1 + exc_f).cumprod()
    x = range(len(nav_m))
    ax1.plot(x, nav_m.values, lw=2, color=C["blue"], label="月度刷新清单")
    ax1.plot(x, nav_f.reindex(nav_m.index).values, lw=2, color=C["orange"],
             label="年度冻结清单")
    for nav, c in [(nav_m, C["blue"]), (nav_f, C["orange"])]:
        ax1.annotate(f"{nav.iloc[-1]:.3f}", (len(nav_m) - 1, nav.iloc[-1]),
                     xytext=(6, 0), textcoords="offset points", color=c,
                     fontweight="bold", va="center")
    ax1.axhline(1, color=C["axis"], lw=0.8)
    years = sorted({m[:4] for m in nav_m.index})
    ticks = [next(i for i, m in enumerate(nav_m.index) if m.startswith(y))
             for y in years]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(years)
    ax1.grid(axis="x", visible=False)
    ax1.set_axisbelow(True)
    ax1.set_title("全市场组合: 剔除低披露尾部10%名单的超额净值 (相对覆盖域等权基准)",
                  loc="left", fontsize=12, pad=12)
    ax1.legend(frameon=False, loc="upper left")

    xs = np.arange(len(yr.index))
    for off, col, c in [(-0.2, "月度刷新", C["blue"]), (0.2, "年度冻结", C["orange"])]:
        v = yr[col].values
        ax2.bar(xs + off, v, 0.36, color=c)
        for xi, vi in zip(xs + off, v):
            ax2.annotate(f"{vi:+.1%}", (xi, vi),
                         xytext=(0, 4 if vi >= 0 else -12),
                         textcoords="offset points", ha="center", fontsize=8,
                         color=C["ink2"])
    ax2.axhline(0, color=C["axis"], lw=0.8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(yr.index)
    ax2.margins(y=0.25)
    ax2.grid(axis="x", visible=False)
    ax2.set_axisbelow(True)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{round(v*100, 2):g}%"))
    ax2.set_title("分年度超额", loc="left", fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "06_fullmarket_enhance.png")
    print(f"\n图表: {CHART_DIR}/06_fullmarket_enhance.png")
