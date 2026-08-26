# -*- coding: utf-8 -*-
"""漂绿测度深化 II: 多评级分歧度调节检验.

思路 (Hu et al. 2023 "The green fog"): 评级机构分歧大 = ESG 信息环境浑浊
= 漂绿温床 → GWS 信号在高分歧组应更强.
分歧度 = 华证/秩鼎/MSCI 三家总分的截面百分位排名的跨机构标准差.

【点时警示】三家评级为当前快照(2026-08), 处于回测窗口(2015-2024)之后;
本检验依赖"评级分歧是公司层面的持久属性"假设(文献支持其高持续性),
结论只能作为暗示性证据; 快照已逐期存档, 一年后可做真点时版.
运行: 需 pyarrow (见 README)
"""
import _path  # noqa: F401
import re
import sys
from pathlib import Path

try:
    import pyarrow  # noqa: F401
except ImportError:
    sys.exit("需要 pyarrow: pip install pyarrow")

import numpy as np
import pandas as pd
from scipy import stats
from greenwash import config
from greenwash.factor import ic_series
from greenwash.pipeline import build_signal_panel

SNAP = config.DATA_EXT / "esg_snapshots"
TAG = "20260817"
MSCI_GRADE = {"AAA": 7, "AA": 6, "A": 5, "BBB": 4, "BB": 3, "B": 2, "CCC": 1}


def num(s):
    """'87.6(AAA)' -> 87.6"""
    m = re.match(r"[\d.]+", str(s))
    return float(m.group()) if m else np.nan


def load_scores():
    hz = pd.read_parquet(SNAP / f"hz_{TAG}.parquet")
    hz["code6"] = hz["股票代码"].str[:6]
    hz = hz.set_index("code6")["ESG评分"].astype(float).rename("hz")

    zd = pd.read_parquet(SNAP / f"zd_{TAG}.parquet")
    zd = zd[zd["股票代码"].str.contains(r"\.(SZ|SH)$", regex=True)]
    zd["code6"] = zd["股票代码"].str[:6]
    zd = zd.set_index("code6")["ESG评分"].map(num).rename("zd")

    ms = pd.read_parquet(SNAP / f"msci_{TAG}.parquet")
    ms = ms[ms["交易市场"] == "CN"]
    ms["code6"] = ms["股票代码"].str[:6]
    ms = ms.set_index("code6")["ESG评分"].map(MSCI_GRADE).rename("msci")

    df = pd.concat([hz, zd, ms], axis=1)
    df = df[~df.index.duplicated()]
    return df


def tstat(s):
    return s.mean() / s.std() * np.sqrt(len(s))


if __name__ == "__main__":
    sc = load_scores()
    ranks = sc.rank(pct=True)
    disag = ranks.std(axis=1).where(sc.notna().sum(axis=1) >= 2)
    print(f"评级覆盖: 华证 {sc.hz.notna().sum()}, 秩鼎 {sc.zd.notna().sum()}, "
          f"MSCI {sc.msci.notna().sum()}; ≥2家 {disag.notna().sum()} 只")
    two = ranks.dropna(subset=["hz", "zd"])
    print(f"机构两两相关(hz-zd): {two.hz.corr(two.zd):.3f}",
          f"(hz-msci): {ranks.hz.corr(ranks.msci):.3f}",
          f"(zd-msci): {ranks.zd.corr(ranks.msci):.3f}")

    d = build_signal_panel(verbose=False)
    d["code6"] = d["code"].str[:6]
    d["disag"] = d["code6"].map(disag)
    dd = d.dropna(subset=["GWS_full", "fwd_ret", "disag"]).copy()
    med = dd.groupby("month")["disag"].transform("median")
    dd["grp"] = np.where(dd["disag"] > med, "高分歧", "低分歧")

    out = {}
    for tag, g in dd.groupby("grp"):
        ics = ic_series(g, "GWS_full", "fwd_ret", "month", min_n=100)["RankIC"]
        tl = g.groupby("month").apply(
            lambda x: x.loc[x["GWS_full"].rank(pct=True) <= 0.1,
                            "fwd_ret"].mean() - x["fwd_ret"].mean()).dropna()
        out[tag] = {"月均只数": g.groupby("month").size().mean(),
                    "RankIC": ics.mean(), "RankICIR": ics.mean() / ics.std(),
                    "t": tstat(ics), "尾部超额(年化)": tl.mean() * 12,
                    "尾部t": stats.ttest_1samp(tl, 0)[0]}
    tab = pd.DataFrame(out).T.round(4)
    print("\n== 评级分歧度调节 (GWS_full, 按当月分歧中位数分组) ==")
    print(tab.to_string())
    tab.to_csv(config.OUT_DIR / "disagreement_moderation.csv",
               encoding="utf-8-sig")
