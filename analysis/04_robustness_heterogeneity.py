# -*- coding: utf-8 -*-
"""稳健性检验与异质性分析 (对应 稳健性检验与异质性分析.do), 并修正两处不合理:

  [修正1] 原 do 的"滞后检验"用 L.ROA 当被解释变量(用当期 GWS 解释过去绩效,
          时序颠倒). 本脚本改为前瞻被解释变量 F.ROA(t 期 GWS 解释 t+1 期 ROA),
          原口径仅作对照保留.
  [修正2] 原 do 的"缩尾对比"只有标题没有实现; 本脚本补上 1%/99% 缩尾基准回归.
"""
import _path  # noqa: F401
import pandas as pd
from greenwash import config
from greenwash.fe import reghdfe, esttab, winsorize

CTRL = ["ESG_per", "Lev", "Dual", "TOP1", "TobinQ"]


def run(df, y, xv, cluster=("industry", "year"), absorb=("industry", "year"),
        label=None, sub=None):
    d = df if sub is None else df[sub]
    return reghdfe(d, y, xv, absorb=list(absorb), cluster=list(cluster),
                   label=label or y)


if __name__ == "__main__":
    df = pd.read_stata(config.PANEL_REG).sort_values(["stkcd", "year"])
    df["ROA_lead1"] = df.groupby("stkcd")["ROA"].shift(-1)   # F.ROA (修正1)
    df["ROA_lag1"] = df.groupby("stkcd")["ROA"].shift(1)     # 原口径对照

    # 1) 替换被解释变量
    ms = [run(df, y, ["GWS_industry"] + CTRL, label=y)
          for y in ["ROA", "Meps", "Mpe", "Mnetpro", "Mebit"]]
    t = esttab(ms)
    print("== 稳健性1: 替换被解释变量 ==\n", t.to_string(), "\n")
    t.to_csv(config.OUT_DIR / "robust_alt_dependent_vars.csv", encoding="utf-8-sig")

    # 2) 替换固定效应
    ms = [run(df, "ROA", ["GWS_industry"] + CTRL, label="industry+year FE"),
          run(df, "ROA", ["GWS_industry"] + CTRL, absorb=["industry"],
              cluster=["industry"], label="industry FE"),
          run(df, "ROA", ["GWS_industry"] + CTRL, absorb=["year"],
              cluster=["year"], label="year FE")]
    print("== 稳健性2: 替换固定效应 ==\n", esttab(ms).to_string(), "\n")

    # 3) 时序检验: 前瞻被解释变量(正确方向) + 原滞后口径对照
    ms = [run(df, "ROA_lead1", ["GWS_industry"] + CTRL, label="F.ROA(修正)"),
          run(df, "ROA_lag1", ["GWS_industry"] + CTRL, label="L.ROA(原do,对照)")]
    t = esttab(ms)
    print("== 稳健性3: 时序检验 ==\n", t.to_string(), "\n")
    t.to_csv(config.OUT_DIR / "robust_forward_roa.csv", encoding="utf-8-sig")

    # 4) 引入/替换 ESG_dis
    ms = [run(df, "ROA", ["GWS_industry", "ESG_dis", "Lev", "Dual", "TOP1",
                          "TobinQ"], label="加入ESG_dis"),
          run(df, "ROA", ["ESG_dis", "Lev", "Dual", "TOP1", "TobinQ"],
              label="仅ESG_dis")]
    print("== 稳健性4: 引入ESG_dis ==\n", esttab(ms).to_string(), "\n")

    # 5) 缩尾对比 (修正2)
    dfw = df.copy()
    for v in ["ROA", "Lev", "TOP1", "TobinQ"]:
        dfw[v] = winsorize(dfw[v])
    ms = [run(df, "ROA", ["GWS_industry"] + CTRL, label="原始"),
          run(dfw, "ROA", ["GWS_industry"] + CTRL, label="缩尾1%/99%")]
    t = esttab(ms)
    print("== 稳健性5: 缩尾对比 ==\n", t.to_string(), "\n")
    t.to_csv(config.OUT_DIR / "robust_winsorized.csv", encoding="utf-8-sig")

    # 6) 异质性: 产权性质
    soe = df["EquityNature"] == "国企"
    ms = [run(df, "ROA", ["GWS_industry"] + CTRL, cluster=["industry"],
              sub=soe, label="国企"),
          run(df, "ROA", ["GWS_industry"] + CTRL, cluster=["industry"],
              sub=~soe, label="非国企")]
    t = esttab(ms)
    print("== 异质性: 产权性质 ==\n", t.to_string(), "\n")
    t.to_csv(config.OUT_DIR / "hetero_ownership.csv", encoding="utf-8-sig")

    # 7) 异质性: 高污染(三种口径)
    ms = []
    for p in ("pollution_1", "pollution_2", "pollution_3"):
        for v, tag in ((1, "高污染"), (0, "非高污染")):
            ms.append(run(df, "ROA", ["GWS_industry"] + CTRL,
                          cluster=["industry"], sub=df[p] == v,
                          label=f"{p}={tag}"))
    t = esttab(ms)
    print("== 异质性: 是否高污染 ==\n", t.to_string())
    t.to_csv(config.OUT_DIR / "hetero_polluting.csv", encoding="utf-8-sig")
