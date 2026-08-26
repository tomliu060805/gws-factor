# -*- coding: utf-8 -*-
"""英文版论文配图(对应 English working paper): figures/v3_paper_en/ 与 paper/figs/.

Fig.1 Orthogonalization steps and predictive power   [both specs]
Fig.2 Decile average next-month returns              [defensive spec]
Fig.3 Cumulative net value                           [defensive spec]
Fig.4 Monthly RankIC series with 12-month moving avg [defensive spec]
Fig.5 Where the silence discount is strongest        [baseline spec, Table 8]
Fig.6 Event-study CAR around penalty announcements   [baseline spec, Table 5]
运行: 见 README 环境说明
"""
import _path  # noqa: F401
import shutil
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

FIG = config.CHART_DIR_EN
FIG.mkdir(parents=True, exist_ok=True)
DEST = config.PAPER_FIG_DIR
DEST.mkdir(parents=True, exist_ok=True)

C = dict(main="#2a78d6", alt="#eb6834", neg="#e34948", ink="#0b0b0b",
         ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
         surface="#ffffff")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "savefig.facecolor": C["surface"], "savefig.dpi": 300,
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
    print(">> building panels ...", flush=True)
    d_base = build_signal_panel(verbose=False)
    d_def = build_signal_panel(verbose=False, start_month=7)
    d_def["GWS_def"] = pd.concat([neutralize(g, "GWS_indyr", STYLES + FUND)
                                  for _, g in d_def.groupby("month")])
    SD = "GWS_def"

    # ---------------- Fig.1 ----------------
    levels = [("Raw gap", "GWS_ind", d_base),
              ("+ Style", "GWS_style", d_base),
              ("+ Style + ESG/\nfundamentals\n(baseline timing)", "GWS_full", d_base),
              ("+ Style + ESG/\nfundamentals\n(defensive timing)", SD, d_def)]
    vals, ts = [], []
    for _, col, dat in levels:
        s = ic_series(dat, col, "fwd_ret", "month", 100)["IC"]
        vals.append(s.mean())
        ts.append(s.mean() / s.std() * np.sqrt(len(s)))
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    xs = np.arange(len(levels))
    ax.bar(xs, vals, 0.6, color=[C["muted"], C["muted"], C["main"], C["main"]])
    for x, v, t in zip(xs, vals, ts):
        ax.annotate(f"{v:+.4f}\n(t={t:.2f})", (x, v),
                    xytext=(0, 6 if v >= 0 else -22), textcoords="offset points",
                    ha="center", fontsize=9, color=C["ink2"])
    ax.axhline(0, color=C["axis"], lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, _, _ in levels], fontsize=8.5)
    ax.set_ylabel("Mean monthly IC")
    ax.margins(y=0.30)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig1.png")

    # ---------------- Fig.2 ----------------
    dd = d_def.dropna(subset=[SD, "fwd_ret"]).copy()
    dd["grp"] = dd.groupby("month")[SD].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    gm = dd.pivot_table(index="month", columns="grp", values="fwd_ret")
    uni = dd.groupby("month")["fwd_ret"].mean()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    means = gm.mean()
    ax.bar(means.index.astype(int), means.values, 0.68,
           color=[C["neg"]] + [C["main"]] * 9)
    ax.axhline(uni.mean(), color=C["ink2"], lw=1.2, ls="--")
    ax.annotate(f"Universe mean {uni.mean():.2%}", (0.55, uni.mean()),
                xytext=(0, 6), textcoords="offset points", fontsize=9,
                color=C["ink2"])
    ax.annotate(f"{means.iloc[0]:.2%}", (1, means.iloc[0]), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=9,
                color=C["neg"], fontweight="bold")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
    ax.set_xlabel("Conditional under-disclosure decile (D1 = most silent)")
    ax.set_ylabel("Mean next-month return")
    clean(ax, pct=True)
    fig.tight_layout()
    fig.savefig(FIG / "fig2.png")

    # ---------------- Fig.3 ----------------
    ls = (gm[10] - gm[1]).dropna()
    g1e = (gm[1] - uni).dropna()
    idx = (pd.PeriodIndex(ls.index, freq="M") + 1).astype(str)
    nav_ls = (1 + ls).cumprod().set_axis(idx)
    nav_g1 = (1 + g1e).cumprod().set_axis(idx)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(range(len(nav_ls)), nav_ls.values, lw=1.8, color=C["main"],
            label="Long-short (D10 $-$ D1)")
    ax.plot(range(len(nav_g1)), nav_g1.values, lw=1.8, color=C["neg"],
            label="Silent decile D1 (vs. comparable firms)")
    for nav, c in ((nav_ls, C["main"]), (nav_g1, C["neg"])):
        ax.annotate(f"{nav.iloc[-1]:.2f}", (len(nav) - 1, nav.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", color=c,
                    fontweight="bold", va="center", fontsize=9)
    ax.axhline(1, color=C["axis"], lw=0.8)
    month_ticks(ax, nav_ls.index)
    ax.set_ylabel("Cumulative value")
    clean(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig3.png")

    # ---------------- Fig.4 ----------------
    ics = ic_series(d_def, SD, "fwd_ret", "month", 100)["RankIC"]
    ics.index = (pd.PeriodIndex(ics.index, freq="M") + 1).astype(str)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(range(len(ics)), ics.values, 0.85,
           color=[C["main"] if v >= 0 else C["neg"] for v in ics], alpha=0.5)
    ax.plot(range(len(ics)), ics.rolling(12).mean().values, lw=1.8,
            color=C["alt"], label="12-month moving average")
    ax.axhline(0, color=C["axis"], lw=0.8)
    ir = ics.mean() / ics.std()
    ax.set_title(f"Mean {ics.mean():+.4f} | RankICIR {ir:.2f} | "
                 f"t = {ir*np.sqrt(len(ics)):.2f} | "
                 f"{(ics>0).mean():.0%} positive months",
                 loc="left", fontsize=9.5, pad=8, color=C["ink2"])
    month_ticks(ax, ics.index)
    ax.set_ylabel("Monthly RankIC")
    clean(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig4.png")

    # ---------------- Fig.5 ----------------
    ds = pd.read_csv(config.OUT_DIR / "barrier_delay_doublesort.csv",
                     index_col=0)
    zh2en = {"价格延迟": "Price delay", "残差波动": "Idio. volatility",
             "非流动性": "Illiquidity", "市值": "Market cap"}
    groups = list(zh2en)
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ys = np.arange(len(groups))[::-1]
    for off, tag, c, lab in ((0.19, "高", C["main"], "High"),
                             (-0.19, "低", C["muted"], "Low")):
        vals = [ds.loc[f"{g}-{tag}", "尾部超额(年化)"] for g in groups]
        tst = [ds.loc[f"{g}-{tag}", "尾部t"] for g in groups]
        ax.barh(ys + off, vals, 0.34, color=c, label=f"{lab} group")
        for y, v, t in zip(ys + off, vals, tst):
            ax.annotate(f"{v:.1%} (t={t:.2f})", (v, y), xytext=(-6, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=8.5, color=C["ink2"])
    ax.set_yticks(ys)
    ax.set_yticklabels([zh2en[g] for g in groups], fontsize=10)
    ax.axvline(0, color=C["axis"], lw=0.8)
    ax.set_xlim(ds["尾部超额(年化)"].min() * 1.55, 0.005)
    ax.set_xlabel("Annualized excess return of the most-silent 10% portfolio")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{round(v*100,2):g}%"))
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig5.png")

    # ---------------- Fig.6 ----------------
    print(">> event study ...", flush=True)
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
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    xs = np.arange(A, B + 1)
    for arr, c, lab in ((hi, C["main"], "High disclosure gap"),
                        (lo, C["muted"], "Low disclosure gap")):
        m = np.vstack(arr).cumsum(axis=1)
        mu, se = m.mean(0), m.std(0) / np.sqrt(len(m))
        ax.plot(xs, mu, lw=1.8, color=c, label=f"{lab} (N={len(m)})")
        ax.fill_between(xs, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.12)
        ax.annotate(f"{mu[-1]:+.2%}", (xs[-1], mu[-1]), xytext=(6, 0),
                    textcoords="offset points", color=c, fontsize=9,
                    fontweight="bold", va="center")
    ax.axvline(0, color=C["axis"], lw=1.0, ls="--")
    ax.axhline(0, color=C["axis"], lw=0.8)
    ax.set_xlabel("Trading days relative to the penalty announcement")
    ax.set_ylabel("Market-adjusted CAR")
    clean(ax, pct=True)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig6.png")

    for i in range(1, 7):
        shutil.copy(FIG / f"fig{i}.png", DEST / f"fig{i}.png")
    print(f"\nEnglish figures -> {FIG}/ and copied to {DEST}/")
