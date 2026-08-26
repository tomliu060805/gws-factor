# -*- coding: utf-8 -*-
"""截面因子检验: IC / ICIR / RankIC / RankICIR / 分组多空.

约定: signal 取第 t 年值, fwd_ret 取第 t+1 年收益(已在调用方对齐),
每年一个截面, 按年计算 IC 序列后汇总:
  IC     = mean(IC_t)          (Pearson)
  ICIR   = mean(IC_t)/std(IC_t)
  RankIC = mean(SpearmanIC_t)  RankICIR 同理
t 统计量 = ICIR * sqrt(T), T 为年数.
"""
import numpy as np
import pandas as pd
from scipy import stats


def ic_series(df, signal, fwd_ret, time="year", min_n=30):
    """逐期截面 Pearson IC 与 Spearman RankIC."""
    rows = []
    for t, g in df[[time, signal, fwd_ret]].dropna().groupby(time):
        if len(g) < min_n:
            continue
        rows.append({
            time: t, "N": len(g),
            "IC": g[signal].corr(g[fwd_ret]),
            "RankIC": g[signal].corr(g[fwd_ret], method="spearman"),
        })
    return pd.DataFrame(rows).set_index(time)


def ic_summary(ics):
    out = {}
    for c in ("IC", "RankIC"):
        s = ics[c]
        T = len(s)
        ir = s.mean() / s.std()
        out[c] = {"mean": s.mean(), "std": s.std(), "IR": ir,
                  "t": ir * np.sqrt(T), "T期数": T,
                  "胜率": (np.sign(s) == np.sign(s.mean())).mean()}
    return pd.DataFrame(out).T


def group_backtest(df, signal, fwd_ret, time="year", n_groups=10):
    """逐年按 signal 分位分组, 等权次年收益; 返回分组均值与多空(G高-G低)统计."""
    d = df[[time, signal, fwd_ret]].dropna().copy()
    d["grp"] = d.groupby(time)[signal].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_groups, labels=False) + 1)
    by_grp = d.groupby("grp")[fwd_ret].agg(["mean", "count"])
    yearly = d.pivot_table(index=time, columns="grp", values=fwd_ret, aggfunc="mean")
    ls = yearly[n_groups] - yearly[1]  # 多空: 高分组 - 低分组
    tstat, p = stats.ttest_1samp(ls, 0)
    ls_stats = pd.Series({
        "年均多空收益": ls.mean(), "年度std": ls.std(), "t": tstat, "p": p,
        "年度胜率": (ls > 0).mean(), "触发年数": len(ls),
        "盈亏比": ls[ls > 0].mean() / abs(ls[ls < 0].mean()) if (ls < 0).any() else np.inf,
        "盈利因子": ls[ls > 0].sum() / abs(ls[ls < 0].sum()) if (ls < 0).any() else np.inf,
    })
    return by_grp, yearly, ls_stats
