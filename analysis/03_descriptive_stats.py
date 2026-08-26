# -*- coding: utf-8 -*-
"""描述性统计 (对应 描述性统计.do), 附 1%/99% 缩尾后对照."""
import _path  # noqa: F401
import pandas as pd
from greenwash import config
from greenwash.fe import winsorize

VARS = ["ROA", "GWS_industry", "ESG_per", "ESG_dis",
        "Lev", "Dual", "TOP1", "TobinQ", "Dturn"]


def tabstat(df):
    t = df[VARS].agg(["count", "mean", "std", "median", "min", "max"]).T
    t.columns = ["N", "均值", "标准差", "中位数", "最小值", "最大值"]
    t["N"] = t["N"].astype(int)
    return t.round(4)


if __name__ == "__main__":
    df = pd.read_stata(config.PANEL_REG)
    raw = tabstat(df)
    print("== 描述性统计(原始) ==\n", raw.to_string())
    dfw = df.copy()
    for v in VARS:
        dfw[v] = winsorize(dfw[v])
    win = tabstat(dfw)
    print("\n== 描述性统计(1%/99% 缩尾后) ==\n", win.to_string())
    raw.to_csv(config.OUT_DIR / "desc_stats.csv", encoding="utf-8-sig")
    win.to_csv(config.OUT_DIR / "desc_stats_winsorized.csv", encoding="utf-8-sig")
