# -*- coding: utf-8 -*-
"""应用层检验: 信号月龄 / 发布时点敏感性 / 指增场景(300 vs 500) / 因子变体.

A. 信号月龄: 年更信号在使用的第 1~12 个月里 RankIC 是否衰减
   (若前几个月集中 → 提频有价值; 若平坦 → 年更不亏)
B. 发布时点敏感性: 信号改为 t+1 年 7 月起应用(保守假设评级年中才可得)
C. 指增场景: 沪深300/中证500 成分内与指数外的 RankIC 与段内尾部效应
D. 因子变体("沉默惩罚"思路): 截面下半区 vs 上半区的 RankIC 不对称性,
   以及披露水平残差因子 DIS_full = ESG_dis ⊥ (风格+行业+ESG_per+基本面)
运行: 需 pyarrow (见 README)
"""
import _path  # noqa: F401
import sys

try:
    import pyarrow  # noqa: F401
except ImportError:
    sys.exit("需要 pyarrow: pip install pyarrow")

import numpy as np
import pandas as pd
from scipy import stats
from greenwash import config
from greenwash.factor import ic_series, ic_summary
from greenwash.pipeline import build_signal_panel, neutralize, STYLES, FUND

W300, W500 = config.W300, config.W500
SIG = "GWS_full"


def monthly_rankic(d, sig=SIG):
    return ic_series(d, sig, "fwd_ret", time="month", min_n=100)["RankIC"]


def tstat(s):
    return s.mean() / s.std() * np.sqrt(len(s))


def membership(path):
    w = pd.read_parquet(path)
    w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
    w["month"] = w["date"].str[:7]
    last = w.groupby("month")["date"].max().rename("last_date")
    ww = w.merge(last, on="month")
    ww = ww[ww["date"] == ww["last_date"]]
    return set(map(tuple, ww[["month", "stock_code"]].values))


if __name__ == "__main__":
    print(">> 构建基准信号面板 (1月起应用) ...", flush=True)
    d1 = build_signal_panel(verbose=False)

    # ---- A. 信号月龄 ----
    ics = monthly_rankic(d1)
    age = pd.DataFrame({"RankIC": ics,
                        "age": ics.index.map(lambda m: int(m[5:7]))})
    by_age = age.groupby("age")["RankIC"].agg(["mean", "count"])
    by_age["t"] = age.groupby("age")["RankIC"].apply(tstat)
    half = age.assign(h=np.where(age["age"] <= 6, "前6个月", "后6个月"))
    by_half = half.groupby("h")["RankIC"].agg(["mean", "count"])
    by_half["t"] = half.groupby("h")["RankIC"].apply(tstat)
    print("\n== A. 信号月龄分组 RankIC ==")
    print(by_age.round(4).to_string())
    print(by_half.round(4).to_string())
    by_age.round(4).to_csv(config.OUT_DIR / "app_signal_age.csv",
                           encoding="utf-8-sig")

    # ---- B. 发布时点敏感性: 7月起应用 ----
    print("\n>> 构建保守时点面板 (7月起应用) ...", flush=True)
    d7 = build_signal_panel(verbose=False, start_month=7)
    cmp = pd.DataFrame({
        "1月起(基准)": ic_summary(ic_series(d1, SIG, "fwd_ret", "month", 100)).loc["RankIC"],
        "7月起(保守)": ic_summary(ic_series(d7, SIG, "fwd_ret", "month", 100)).loc["RankIC"],
    }).T
    print("\n== B. 发布时点敏感性 (GWS_full RankIC) ==")
    print(cmp.round(4).to_string())
    cmp.round(4).to_csv(config.OUT_DIR / "app_timing_sensitivity.csv",
                        encoding="utf-8-sig")

    # ---- C. 指增场景 ----
    m300, m500 = membership(W300), membership(W500)
    dd = d1.dropna(subset=[SIG, "fwd_ret"]).copy()
    key = list(zip(dd["month"], dd["code"]))
    in3 = pd.Series([k in m300 for k in key], index=dd.index)
    in5 = pd.Series([k in m500 for k in key], index=dd.index)
    dd["seg"] = np.where(in3, "沪深300内", np.where(in5, "中证500内", "指数外"))
    rows = {}
    for seg, g in dd.groupby("seg"):
        ics_s = monthly_rankic(g)
        tl = g.groupby("month").apply(
            lambda x: x.loc[x[SIG].rank(pct=True) <= 0.1, "fwd_ret"].mean()
            - x["fwd_ret"].mean()).dropna()
        rows[seg] = {"月均只数": g.groupby("month").size().mean(),
                     "RankIC": ics_s.mean(), "RankICIR": ics_s.mean() / ics_s.std(),
                     "t": tstat(ics_s), "后10%尾部超额(年化)": tl.mean() * 12,
                     "尾部t": stats.ttest_1samp(tl, 0)[0]}
    seg_tab = pd.DataFrame(rows).T
    print("\n== C. 指增场景: 成分内 RankIC 与段内尾部效应 ==")
    print(seg_tab.round(4).to_string())
    seg_tab.round(4).to_csv(config.OUT_DIR / "app_index_segments.csv",
                            encoding="utf-8-sig")

    # ---- D. 因子变体 ----
    # D1: 截面上/下半区 RankIC (不对称性: 信息是否集中在披露不足侧)
    def half_ic(g, side):
        r = g[SIG].rank(pct=True)
        sub = g[r <= 0.5] if side == "low" else g[r > 0.5]
        return (sub[SIG].corr(sub["fwd_ret"], method="spearman")
                if len(sub) > 50 else np.nan)
    lo = dd.groupby("month").apply(half_ic, side="low").dropna()
    hi = dd.groupby("month").apply(half_ic, side="high").dropna()
    print("\n== D1. 截面半区 RankIC (GWS_full) ==")
    print(f"下半区(披露不足侧): {lo.mean():+.4f}  IR {lo.mean()/lo.std():.2f}  t={tstat(lo):.2f}")
    print(f"上半区(披露过度侧): {hi.mean():+.4f}  IR {hi.mean()/hi.std():.2f}  t={tstat(hi):.2f}")

    # D2: 披露水平残差因子 (直接检验"沉默惩罚", 不经 GWS 构造)
    d1["DIS_full"] = pd.concat([neutralize(g, "ESG_dis", STYLES + FUND)
                                for _, g in d1.groupby("month")])
    summ = {}
    for s in (SIG, "DIS_full"):
        summ[s] = ic_summary(ic_series(d1, s, "fwd_ret", "month", 100))
    dis_tab = pd.concat(summ, names=["signal", "type"]).round(4)
    print("\n== D2. 披露残差因子 vs GWS_full ==")
    print(dis_tab.to_string())
    dis_tab.to_csv(config.OUT_DIR / "app_factor_variants.csv",
                   encoding="utf-8-sig")
    corr = d1[[SIG, "DIS_full"]].dropna()
    print(f"\ncorr(GWS_full, DIS_full) = {corr[SIG].corr(corr['DIS_full']):.3f}")
