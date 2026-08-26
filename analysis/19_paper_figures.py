# -*- coding: utf-8 -*-
"""论文配图: 生成 图1—图6 至 figures/v2_paper_cn/.

图1 正交化层次与预测力(方法要点)      [两口径对照]
图2 十分组月均前瞻收益(单侧形态)      [防御口径]
图3 累计净值: 多空与沉默组相对可比公司  [防御口径]
图4 月度RankIC序列与12月滚动均值       [防御口径]
图5 双重排序: 沉默折价在何种环境最强    [基准口径, 对应正文表8]
图6 处罚公告事件研究CAR路径            [基准口径, 对应正文表5]
期刊配图取向: 克制配色(单色系+对照色)、直接标注、灰度可读.
运行: 见 README 环境说明
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

from greenwash import config
from greenwash.factor import ic_series
from greenwash.marketdata import build_daily_returns, market_returns
from greenwash.pipeline import (build_signal_panel, neutralize, STYLES, FUND)

FIG = config.ROOT / "figures" / "v2_paper_cn"
FIG.mkdir(parents=True, exist_ok=True)

C = dict(main="#2a78d6", alt="#eb6834", neg="#e34948", ink="#0b0b0b",
         ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
         surface="#ffffff")
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "savefig.facecolor": C["surface"], "savefig.dpi": 200,
    "axes.edgecolor": C["axis"], "axes.linewidth": 0.8,
    "xtick.color": C["muted"], "ytick.color": C["muted"],
    "axes.labelcolor": C["ink2"], "text.color": C["ink"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def clean(ax, pct=False):
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    if pct:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{round(v*100, 2):g}%"))


def month_ticks(ax, idx):
    years = sorted({m[:4] for m in idx})
    ax.set_xticks([next(i for i, m in enumerate(idx) if m.startswith(y))
                   for y in years])
    ax.set_xticklabels(years)


if __name__ == "__main__":
    print(">> 构建两口径面板 ...", flush=True)
    d_base = build_signal_panel(verbose=False)
    d_def = build_signal_panel(verbose=False, start_month=7)
    d_def["GWS_def"] = pd.concat([neutralize(g, "GWS_indyr", STYLES + FUND)
                                  for _, g in d_def.groupby("month")])
    SD = "GWS_def"

    # ---------------- 图1 正交化层次 ----------------
    levels = [("原始裂口", "GWS_ind", d_base), ("+风格中性", "GWS_style", d_base),
              ("+风格+ESG/基本面\n(基准口径)", "GWS_full", d_base),
              ("+风格+ESG/基本面\n(防御口径)", SD, d_def)]
    vals, ts = [], []
    for _, col, dat in levels:
        s = ic_series(dat, col, "fwd_ret", "month", 100)["IC"]
        vals.append(s.mean())
        ts.append(s.mean() / s.std() * np.sqrt(len(s)))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    xs = np.arange(len(levels))
    cols = [C["muted"], C["muted"], C["main"], C["main"]]
    ax.bar(xs, vals, 0.6, color=cols)
    for x, v, t in zip(xs, vals, ts):
        ax.annotate(f"{v:+.4f}\n(t={t:.2f})", (x, v),
                    xytext=(0, 6 if v >= 0 else -22), textcoords="offset points",
                    ha="center", fontsize=9, color=C["ink2"])
    ax.axhline(0, color=C["axis"], lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, _, _ in levels], fontsize=9)
    ax.set_ylabel("月度IC均值")
    ax.margins(y=0.28)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_orthogonalization.png")

    # ---------------- 图2 十分组 ----------------
    dd = d_def.dropna(subset=[SD, "fwd_ret"]).copy()
    dd["grp"] = dd.groupby("month")[SD].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    gm = dd.pivot_table(index="month", columns="grp", values="fwd_ret")
    uni = dd.groupby("month")["fwd_ret"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    means = gm.mean()
    bars = ax.bar(means.index.astype(int), means.values, 0.68,
                  color=[C["neg"]] + [C["main"]] * 9)
    ax.axhline(uni.mean(), color=C["ink2"], lw=1.2, ls="--")
    ax.annotate(f"全样本均值 {uni.mean():.2%}", (0.55, uni.mean()),
                xytext=(0, 6), textcoords="offset points", fontsize=9,
                color=C["ink2"])
    ax.annotate(f"{means.iloc[0]:.2%}", (1, means.iloc[0]), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=9,
                color=C["neg"], fontweight="bold")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"G{i}" for i in range(1, 11)])
    ax.set_xlabel("条件性披露不足十分位组(G1=最沉默)")
    ax.set_ylabel("次月收益均值")
    clean(ax, pct=True)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_decile_returns.png")

    # ---------------- 图3 净值 ----------------
    ls = (gm[10] - gm[1]).dropna()
    g1e = (gm[1] - uni).dropna()
    idx = (pd.PeriodIndex(ls.index, freq="M") + 1).astype(str)
    nav_ls = (1 + ls).cumprod().set_axis(idx)
    nav_g1 = (1 + g1e).cumprod().set_axis(idx)
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    ax.plot(range(len(nav_ls)), nav_ls.values, lw=2, color=C["main"],
            label="多空组合(G10−G1)")
    ax.plot(range(len(nav_g1)), nav_g1.values, lw=2, color=C["neg"],
            label="沉默组G1(相对可比公司)")
    for nav, c in ((nav_ls, C["main"]), (nav_g1, C["neg"])):
        ax.annotate(f"{nav.iloc[-1]:.2f}", (len(nav) - 1, nav.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", color=c,
                    fontweight="bold", va="center", fontsize=9)
    ax.axhline(1, color=C["axis"], lw=0.8)
    month_ticks(ax, nav_ls.index)
    ax.set_ylabel("累计净值")
    clean(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_nav.png")

    # ---------------- 图4 RankIC 序列 ----------------
    ics = ic_series(d_def, SD, "fwd_ret", "month", 100)["RankIC"]
    ics.index = (pd.PeriodIndex(ics.index, freq="M") + 1).astype(str)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.bar(range(len(ics)), ics.values, 0.85,
           color=[C["main"] if v >= 0 else C["neg"] for v in ics], alpha=0.5)
    ax.plot(range(len(ics)), ics.rolling(12).mean().values, lw=2,
            color=C["alt"], label="12个月滚动均值")
    ax.axhline(0, color=C["axis"], lw=0.8)
    ir = ics.mean() / ics.std()
    ax.set_title(f"均值 {ics.mean():+.4f} · RankICIR {ir:.2f} · "
                 f"t={ir*np.sqrt(len(ics)):.2f} · 胜率 {(ics>0).mean():.0%}",
                 loc="left", fontsize=10, pad=8, color=C["ink2"])
    month_ticks(ax, ics.index)
    ax.set_ylabel("月度RankIC")
    clean(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_rankic_series.png")

    # ---------------- 图5 双重排序(对应表8) ----------------
    ds = pd.read_csv(config.OUT_DIR / "barrier_delay_doublesort.csv",
                     index_col=0)
    groups = ["价格延迟", "残差波动", "非流动性", "市值"]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ys = np.arange(len(groups))[::-1]
    for off, tag, c in ((0.19, "高", C["main"]), (-0.19, "低", C["muted"])):
        vals = [ds.loc[f"{g}-{tag}", "尾部超额(年化)"] for g in groups]
        tst = [ds.loc[f"{g}-{tag}", "尾部t"] for g in groups]
        ax.barh(ys + off, vals, 0.34, color=c, label=f"{tag}分组")
        for y, v, t in zip(ys + off, vals, tst):
            ax.annotate(f"{v:.1%} (t={t:.2f})", (v, y), xytext=(-6, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=8.5, color=C["ink2"])
    ax.set_yticks(ys)
    ax.set_yticklabels(groups, fontsize=10)
    ax.axvline(0, color=C["axis"], lw=0.8)
    ax.set_xlabel("最沉默10%组合的年化超额收益")
    ax.set_xlim(ds["尾部超额(年化)"].min() * 1.55, 0.005)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{round(v*100,2):g}%"))
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_double_sort.png")

    # ---------------- 图6 事件研究 CAR 路径 ----------------
    print(">> 事件研究 CAR 路径 ...", flush=True)
    pen = pd.read_parquet(config.DATA_EXT / "cninfo_penalty_2014_2024.parquet")
    pen["date"] = pd.to_datetime(pen["公告时间"], format="mixed").dt.normalize()
    pen["month"] = pen["date"].dt.strftime("%Y-%m")
    pen["code"] = pen["symbol"].map(
        lambda s: (s + ".XSHG") if str(s)[0] == "6" else (s + ".XSHE"))
    ret, _ = build_daily_returns()
    mkt = market_returns(ret.index)
    ar = ret.sub(mkt, axis=0)
    dates = ret.index
    look = d_base.set_index(["code", "month"])["GWS_full"]
    med = d_base.groupby("month")["GWS_full"].median()
    A, B = -5, 20
    hi, lo = [], []
    for _, r in pen.iterrows():
        pm = (pd.Period(r["month"], "M") - 1).strftime("%Y-%m")
        s = look.get((r["code"], pm), np.nan)
        if np.isnan(s) or r["code"] not in ar.columns or pm not in med:
            continue
        pos = dates.searchsorted(r["date"])
        if pos + A < 0 or pos + B >= len(dates):
            continue
        w = ar[r["code"]].iloc[pos + A: pos + B + 1]
        if w.notna().sum() <= (B - A) * 0.7:
            continue
        (hi if s > med[pm] else lo).append(w.fillna(0).values)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = np.arange(A, B + 1)
    for arr, c, lab in ((hi, C["main"], "高披露裂口"), (lo, C["muted"], "低披露裂口")):
        m = np.vstack(arr).cumsum(axis=1)
        mu, se = m.mean(0), m.std(0) / np.sqrt(len(m))
        ax.plot(xs, mu, lw=2, color=c, label=f"{lab}(N={len(m)})")
        ax.fill_between(xs, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.12)
        ax.annotate(f"{mu[-1]:+.2%}", (xs[-1], mu[-1]), xytext=(6, 0),
                    textcoords="offset points", color=c, fontsize=9,
                    fontweight="bold", va="center")
    ax.axvline(0, color=C["axis"], lw=1.0, ls="--")
    ax.axhline(0, color=C["axis"], lw=0.8)
    ax.set_xlabel("相对处罚公告日的交易日")
    ax.set_ylabel("市场调整累计异常收益(CAR)")
    clean(ax, pct=True)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_event_car.png")

    print(f"\n六张配图已输出至 {FIG}/")
