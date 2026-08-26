# -*- coding: utf-8 -*-
"""GWS 作为低频(年度)截面因子的检验 —— 原 Stata 项目没有的量化视角延伸.

信号: 第 t 年 GWS(三种口径) 及 ESG_per / ESG_dis 对照
收益: 第 t+1 年个股年度超额收益 Rf (仅在 t 与 t+1 连续时配对, 避免跨缝)
输出:
  1) 逐年 IC / RankIC 序列与汇总 (IC, ICIR, RankIC, RankICIR, t, 胜率)
  2) 增量检验: GWS 逐年截面残差化于 [ESG_per, log资产, Lev, ROA, mb] 后的 IC
     —— 回答"控制 ESG 水平与基本面后是否还有增量信息"
  3) Fama-MacBeth 截面回归 (fwd_ret ~ GWS + 基本面控制)
  4) 十分组多空回测(含年度胜率/盈亏比/盈利因子)
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from greenwash import config, gws
from greenwash.factor import ic_series, ic_summary, group_backtest

FUND = ["ESG_per", "log_asset", "Lev", "ROA", "mb"]  # 基本面中性化基


def residualize_by_year(df, signal, controls):
    """逐年截面 OLS 残差 (含常数项)."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("year"):
        sub = g[[signal] + controls].dropna()
        if len(sub) < 50:
            continue
        Xm = np.column_stack([np.ones(len(sub)), sub[controls].to_numpy(float)])
        beta = np.linalg.lstsq(Xm, sub[signal].to_numpy(float), rcond=None)[0]
        out.loc[sub.index] = sub[signal].to_numpy(float) - Xm @ beta
    return out


def fama_macbeth(df, y, xvars):
    """逐年截面回归, 返回各系数的 FM 均值/t."""
    rows = []
    for t, g in df.groupby("year"):
        sub = g[[y] + xvars].dropna()
        if len(sub) < 50:
            continue
        Xm = np.column_stack([np.ones(len(sub)), sub[xvars].to_numpy(float)])
        beta = np.linalg.lstsq(Xm, sub[y].to_numpy(float), rcond=None)[0]
        rows.append(dict(zip(["const"] + xvars, beta), year=t))
    b = pd.DataFrame(rows).set_index("year")
    T = len(b)
    return pd.DataFrame({"FM系数": b.mean(),
                         "t": b.mean() / b.std() * np.sqrt(T),
                         "T期数": T})


if __name__ == "__main__":
    df = pd.read_stata(config.PANEL_FULL)
    df = gws.add_gws(df).sort_values(["stkcd", "year"])
    df["log_asset"] = np.log(df["asset"].where(df["asset"] > 0))

    # t+1 年超额收益, 仅连续年份配对 (annual_excess_ret 列为坏数据, 弃用)
    g = df.groupby("stkcd")
    df["fwd_ret"] = g["Rf"].shift(-1)
    df.loc[g["year"].shift(-1) != df["year"] + 1, "fwd_ret"] = np.nan

    # ---- 1) 原始 IC ----
    signals = ["GWS_ind", "GWS_indyr", "GWS_yr", "ESG_per", "ESG_dis"]
    summ = {}
    for s in signals:
        ics = ic_series(df, s, "fwd_ret")
        summ[s] = ic_summary(ics)
        if s == "GWS_ind":
            print("== GWS_ind 逐年 IC ==\n", ics.round(4).to_string(), "\n")
    raw_tab = pd.concat(summ, names=["signal", "type"]).round(4)
    print("== 原始 IC 汇总 (IC/ICIR/RankIC/RankICIR) ==\n", raw_tab.to_string(), "\n")
    raw_tab.to_csv(config.OUT_DIR / "factor_ic_raw.csv", encoding="utf-8-sig")

    # ---- 2) 增量检验: 基本面中性化后的 IC ----
    neu = {}
    for s in ["GWS_ind", "GWS_indyr", "ESG_dis"]:
        df[f"{s}_neu"] = residualize_by_year(df, s, FUND)
        neu[f"{s}_neu"] = ic_summary(ic_series(df, f"{s}_neu", "fwd_ret"))
    neu_tab = pd.concat(neu, names=["signal", "type"]).round(4)
    print("== 中性化增量 IC (残差化于 ESG_per+log资产+Lev+ROA+mb) ==")
    print(neu_tab.to_string(), "\n")
    neu_tab.to_csv(config.OUT_DIR / "factor_ic_neutralized.csv", encoding="utf-8-sig")

    # ---- 3) Fama-MacBeth ----
    fm = fama_macbeth(df, "fwd_ret", ["GWS_ind"] + FUND)
    print("== Fama-MacBeth: fwd_ret ~ GWS_ind + 基本面 ==\n",
          fm.round(4).to_string(), "\n")
    fm.round(6).to_csv(config.OUT_DIR / "factor_fama_macbeth.csv",
                       encoding="utf-8-sig")

    # ---- 4) 十分组多空 ----
    by_grp, yearly, ls = group_backtest(df, "GWS_ind", "fwd_ret", n_groups=10)
    print("== GWS_ind 十分组: 次年超额收益均值 ==\n",
          by_grp.round(4).to_string())
    print("\n== 多空(G10-G1)统计 ==\n", ls.round(4).to_string())
    yearly.round(4).to_csv(config.OUT_DIR / "factor_decile_yearly.csv",
                           encoding="utf-8-sig")
