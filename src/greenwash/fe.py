# -*- coding: utf-8 -*-
"""固定效应回归: 纯 statsmodels 实现 Stata reghdfe / xtreg,fe 的等价功能.

- absorb 多维固定效应: 类别哑变量精确吸收 (点估计与 reghdfe 完全一致)
- vce(cluster g) / vce(cluster g1 g2): 一维/二维(CGM)聚类稳健标准误
  注: 二维聚类在簇数较少(如年份~10)时可能出现非正定修正, statsmodels 会对
  个别哑变量系数给出 NaN 标准误, 核心解释变量不受影响; 结果表中若出现 NaN
  会显式标出而不是静默.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


def winsorize(s, lower=0.01, upper=0.99):
    """按分位数缩尾 (对应 Stata winsor2)."""
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def reghdfe(df, y, xvars, absorb=(), cluster=(), label=None):
    """等价于: reghdfe y xvars, absorb(absorb) vce(cluster cluster)

    返回 dict: {label, res, table(仅 xvars+const), N, ar2, data}
    """
    absorb, cluster = list(absorb), list(cluster)
    need = [y] + list(xvars) + absorb + [c for c in cluster if c not in absorb]
    d = df[need].dropna().copy()

    X = d[list(xvars)].astype(float)
    for a in absorb:
        dum = pd.get_dummies(d[a], prefix=f"FE_{a}", drop_first=True, dtype=float)
        X = pd.concat([X, dum], axis=1)
    X = sm.add_constant(X)

    if len(cluster) == 1:
        fitkw = dict(cov_type="cluster",
                     cov_kwds={"groups": d[cluster[0]].values})
    elif len(cluster) == 2:
        groups = np.column_stack([pd.factorize(d[c])[0] for c in cluster])
        fitkw = dict(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        fitkw = dict(cov_type="HC1")  # 无聚类时用异方差稳健(对应 Stata 的 r)

    with warnings.catch_warnings():
        # 二维聚类对部分哑变量可能出现负方差(CGM 非正定), 只影响 FE 哑变量
        warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
        res = sm.OLS(d[y].astype(float), X).fit(**fitkw)
        rows = [c for c in list(xvars) + ["const"] if c in res.params.index]
        table = pd.DataFrame({
            "coef": res.params[rows], "se": res.bse[rows],
            "t": res.tvalues[rows], "p": res.pvalues[rows],
        })
    if table[["se", "t", "p"]].isna().any().any():
        warnings.warn(f"[{label or y}] 核心变量标准误出现 NaN, 请改用一维聚类")
    return {"label": label or y, "res": res, "table": table,
            "N": int(res.nobs), "ar2": res.rsquared_adj, "data": d}


def xtreg_fe(df, y, xvars, panel="stkcd", time="year", cluster=None):
    """等价于: xtset panel time; xtreg y xvars i.time, fe r
    (个体+时间固定效应; Stata 面板的 robust 即按个体聚类)"""
    return reghdfe(df, y, xvars, absorb=[panel, time],
                   cluster=[cluster or panel])


def esttab(models, digits=4):
    """等价于 esttab: 多模型并排系数表(* p<0.1 ** p<0.05 *** p<0.01)"""
    def star(p):
        if np.isnan(p):
            return "(NaN)"
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    cols = {}
    for m in models:
        t = m["table"]
        col = {}
        for v in t.index:
            col[v] = f"{t.loc[v, 'coef']:.{digits}f}{star(t.loc[v, 'p'])}"
            col[v + "  (t)"] = f"({t.loc[v, 't']:.2f})"
        col["N"] = str(m["N"])
        col["adj.R2"] = f"{m['ar2']:.3f}"
        cols[m["label"]] = col
    order = []
    for m in models:
        for v in m["table"].index:
            if v == "const":
                continue
            for k in (v, v + "  (t)"):
                if k not in order:
                    order.append(k)
    order += ["const", "const  (t)", "N", "adj.R2"]
    return pd.DataFrame(cols).reindex(order).fillna("")
