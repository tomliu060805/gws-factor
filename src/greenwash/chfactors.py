# -*- coding: utf-8 -*-
"""中国三因子 (Liu, Stambaugh & Yuan 2019, JFE) 与经典因子的自建实现.

CH-3 关键设定(与 FF3 的区别):
  - 剔除市值最小 30% 的股票 —— 中国壳价值污染小市值组合;
  - 价值指标用 EP = 盈利/价格(而非 B/M), LSY 证明 EP 在中国最有效;
  - 2×3 独立排序(规模中位数 × EP 的 30/40/30), 组合市值加权.
    SMB = 平均(S/V,S/M,S/G) − 平均(B/V,B/M,B/G)
    VMG = 平均(S/V,B/V)     − 平均(S/G,B/G)
  - MKT = 全样本(剔小30%后)市值加权收益 − 无风险利率.
另提供 FF3 口径(不剔小票, 价值用 B/M)作为对照.
无风险利率: CSMAR 月度化无风险利率(见 config.RF_FILE).
"""
import numpy as np
import pandas as pd

from . import config
from .marketdata import build_monthly_panel

RF_FILE = config.RF_FILE


def risk_free():
    """月度无风险利率(小数), index = 'YYYY-MM'."""
    rf = pd.read_excel(RF_FILE, skiprows=[1, 2])
    rf = rf.rename(columns={rf.columns[1]: "date", rf.columns[3]: "rfm"})
    rf = rf[pd.to_datetime(rf["date"], errors="coerce").notna()]
    rf["month"] = pd.to_datetime(rf["date"]).dt.strftime("%Y-%m")
    rf["rfm"] = pd.to_numeric(rf["rfm"], errors="coerce") / 100.0
    return rf.groupby("month")["rfm"].last()


def _vw(g):
    w = g["MCAP"]
    return (g["ret"] * w).sum() / w.sum() if w.sum() > 0 else np.nan


def build_factors(panel=None, ch3=True):
    """返回月度因子表: MKT / SMB / VMG(或HML) , index='YYYY-MM'(收益所属月)."""
    p = build_monthly_panel() if panel is None else panel
    val_col = "EP" if ch3 else "BTOP"
    d = p.dropna(subset=["ret", "MCAP", val_col]).copy()

    # 组合按上月末特征构建, 持有当月 → 用滞后特征
    d = d.sort_values(["code", "month"])
    for c in ("MCAP", val_col):
        d[f"lag_{c}"] = d.groupby("code")[c].shift(1)
    d["lag_month"] = d.groupby("code")["month"].shift(1)
    # 仅保留相邻月配对
    gap = (pd.PeriodIndex(d["month"], freq="M").astype("int64")
           - pd.PeriodIndex(d["lag_month"].fillna(d["month"]),
                            freq="M").astype("int64"))
    ok = gap == 1
    d = d[ok].dropna(subset=[f"lag_MCAP", f"lag_{val_col}"])
    d["MCAP"] = d["lag_MCAP"]          # 加权用上月末市值

    rows = {}
    for m, g in d.groupby("month"):
        if len(g) < 200:
            continue
        if ch3:                         # 剔除最小 30%
            g = g[g["lag_MCAP"] > g["lag_MCAP"].quantile(0.30)]
        if len(g) < 100:
            continue
        size_hi = g["lag_MCAP"] > g["lag_MCAP"].median()
        v_lo, v_hi = g[f"lag_{val_col}"].quantile([0.30, 0.70])
        vg = np.where(g[f"lag_{val_col}"] >= v_hi, "V",
                      np.where(g[f"lag_{val_col}"] <= v_lo, "G", "M"))
        g = g.assign(_s=np.where(size_hi, "B", "S"), _v=vg)
        port = {k: _vw(sub) for k, sub in g.groupby(["_s", "_v"])}
        need = [("S", "V"), ("S", "M"), ("S", "G"),
                ("B", "V"), ("B", "M"), ("B", "G")]
        if any(k not in port or np.isnan(port[k]) for k in need):
            continue
        smb = np.mean([port[("S", v)] for v in "VMG"]) - \
            np.mean([port[("B", v)] for v in "VMG"])
        vmg = np.mean([port[(s, "V")] for s in "SB"]) - \
            np.mean([port[(s, "G")] for s in "SB"])
        rows[m] = {"MKT_raw": _vw(g), "SMB": smb,
                   ("VMG" if ch3 else "HML"): vmg}
    f = pd.DataFrame(rows).T
    rf = risk_free().reindex(f.index)
    f["MKT"] = f["MKT_raw"] - rf
    f["RF"] = rf
    return f.drop(columns="MKT_raw")


def alpha_table(port_ret, factors, name="组合", excess=True):
    """组合月度收益对因子回归, 返回 alpha(年化)/t 与因子暴露(Newey-West 6期)."""
    import statsmodels.api as sm
    df = pd.concat([port_ret.rename("y"), factors], axis=1).dropna()
    y = df["y"] - (df["RF"] if excess else 0.0)
    specs = {"CAPM": ["MKT"], "三因子": ["MKT", "SMB",
                                     "VMG" if "VMG" in df else "HML"]}
    out = {}
    for k, xs in specs.items():
        X = sm.add_constant(df[xs])
        r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        out[k] = {"alpha(年化)": r.params["const"] * 12,
                  "alpha_t": r.tvalues["const"], **{
                      f"beta_{x}": r.params[x] for x in xs},
                  "R2": r.rsquared, "月数": int(r.nobs)}
    return pd.DataFrame(out).T.round(4)
