# -*- coding: utf-8 -*-
"""GWS 构造 + 与原 Stata 结果对拍.

对应 .do: 计算漂绿值+模型一代码.do / 按行业标准化… / 按行业年度同时标准化…
"""
import _path  # noqa: F401
import pandas as pd
from greenwash import config, gws

if __name__ == "__main__":
    df = pd.read_stata(config.PANEL_FULL)
    df = gws.add_gws(df)

    print("== 与 Stata 存储值对拍 ==")
    pairs = [("GWS_ind", "GWS_industry", "行业口径(应=1)"),
             ("GWS_yr", "GWS_byyear", "年度口径(Stata为早期子样本所算,略有出入)"),
             ("GWS_indyr", "GWS_industryyear", "行业×年口径(Stata含bug,应<1)")]
    for py, st, note in pairs:
        b = df[[py, st]].dropna()
        print(f"{st:20s} corr={b[py].corr(b[st]):.6f}  "
              f"max|diff|={(b[py]-b[st]).abs().max():.2e}   {note}")

    bug = gws.replicate_stata_bug(df)
    b = pd.concat([bug.rename("bug"), df["GWS_industryyear"]], axis=1).dropna()
    print(f"{'bug复现口径':18s} corr={b['bug'].corr(b['GWS_industryyear']):.6f}  "
          f"(=1 即证明原Stata标准差漏了按年分组)")

    print("\n== 三种口径 GWS 描述统计(正确口径) ==")
    print(df[["GWS_ind", "GWS_indyr", "GWS_yr"]].describe().round(4).to_string())
    df[["stkcd", "year", "GWS_ind", "GWS_indyr", "GWS_yr"]].to_csv(
        config.OUT_DIR / "gws_scores.csv", index=False, encoding="utf-8-sig")
