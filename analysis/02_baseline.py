# -*- coding: utf-8 -*-
"""模型一(基准回归): 逐步加入控制变量.

对应 .do: 计算漂绿值+模型一代码.do / 画图代码.do
(a) reghdfe ROA GWS…, absorb(industry year) vce(cluster industry)  — 论文主回归
(b) xtreg Rf …, fe r — 年度超额收益版(原项目弃用: GWS 不显著, 如实保留)
"""
import _path  # noqa: F401
import pandas as pd
from greenwash import config
from greenwash.fe import reghdfe, xtreg_fe, esttab

if __name__ == "__main__":
    df = pd.read_stata(config.PANEL_REG)
    specs = [
        ["GWS_industry"],
        ["GWS_industry", "ESG_per"],
        ["GWS_industry", "ESG_per", "Lev"],
        ["GWS_industry", "ESG_per", "Lev", "Dual"],
        ["GWS_industry", "ESG_per", "Lev", "Dual", "TOP1", "TobinQ"],
    ]
    models = [reghdfe(df, "ROA", xv, absorb=["industry", "year"],
                      cluster=["industry"], label=f"({i+1}) ROA")
              for i, xv in enumerate(specs)]
    tab = esttab(models)
    print("== (a) 主回归: reghdfe ROA, absorb(industry year) cluster(industry) ==")
    print(tab.to_string())
    tab.to_csv(config.OUT_DIR / "panel_baseline_stepwise.csv", encoding="utf-8-sig")

    df1 = pd.read_stata(config.PANEL_FULL)
    m = xtreg_fe(df1, "Rf",
                 ["ESG_per", "GWS_industry", "asset", "Lev", "ROA", "mb",
                  "TobinQ", "Dual", "Dturn", "cashflow", "TOP1"])
    m["label"] = "Rf(超额收益)"
    print("\n== (b) xtreg Rf …, fe r (GWS 不显著, 原项目由此转向 ROA) ==")
    print(esttab([m]).to_string())
