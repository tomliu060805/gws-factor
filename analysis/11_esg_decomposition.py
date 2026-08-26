# -*- coding: utf-8 -*-
"""漂绿测度深化 I: E/S/G 分项裂口分解 + 高污染行业交互.

动机: "洗绿/漂绿"的本义是环境(E)维度的说做不一.
  - 若 E 裂口有 alpha 而 G 裂口(安慰剂)没有 → 信号确为"绿色包装"行为;
  - 若三分项同强 → 信号实为一般性"披露风格/治理"信号.
交互: 漂绿动机在高污染行业最强(监管压力大、洗白收益高), 检验 E 裂口
  在 高污染 vs 非高污染 (三种口径) 的 RankIC 与尾部效应.
分项裂口构造: GWS{k}_ind = z(彭博{k}分项披露|行业) − z(ESG表现|行业),
  再经风格+行业+基本面完全中性化 → GWS{k}_full.
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
from greenwash.pipeline import build_signal_panel

SIGS = ["GWS_full", "GWSE_full", "GWSS_full", "GWSG_full"]
LAB = {"GWS_full": "总分裂口", "GWSE_full": "E裂口(环境)",
       "GWSS_full": "S裂口(社会)", "GWSG_full": "G裂口(治理·安慰剂)"}


def tstat(s):
    return s.mean() / s.std() * np.sqrt(len(s))


if __name__ == "__main__":
    print(">> 构建含分项的信号面板 ...", flush=True)
    d = build_signal_panel(verbose=False, with_sub=True)

    # ---- 1. 分项 IC 对比 ----
    summ = {LAB[s]: ic_summary(ic_series(d, s, "fwd_ret", "month", 100))
            for s in SIGS}
    tab = pd.concat(summ, names=["signal", "type"]).round(4)
    print("\n== E/S/G 分项裂口 月频 IC ==")
    print(tab.to_string())
    tab.to_csv(config.OUT_DIR / "decomp_esg_ic.csv", encoding="utf-8-sig")

    corr = d[SIGS].dropna().corr().round(3)
    print("\n分项因子相关矩阵:\n", corr.to_string())

    # 分项尾部效应 (后10% vs 全样本)
    rows = {}
    for s in SIGS:
        g = d.dropna(subset=[s, "fwd_ret"])
        tl = g.groupby("month").apply(
            lambda x: x.loc[x[s].rank(pct=True) <= 0.1, "fwd_ret"].mean()
            - x["fwd_ret"].mean()).dropna()
        rows[LAB[s]] = {"尾部超额(年化)": tl.mean() * 12,
                        "t": stats.ttest_1samp(tl, 0)[0]}
    print("\n分项尾部效应:\n", pd.DataFrame(rows).T.round(4).to_string())

    # ---- 2. 高污染交互 ----
    out = {}
    for s in ("GWS_full", "GWSE_full"):
        for p in ("pollution_1", "pollution_2", "pollution_3"):
            for v, tag in ((1, "高污染"), (0, "非高污染")):
                g = d[d[p] == v].dropna(subset=[s, "fwd_ret"])
                ics = ic_series(g, s, "fwd_ret", "month", min_n=50)["RankIC"]
                tl = g.groupby("month").apply(
                    lambda x: x.loc[x[s].rank(pct=True) <= 0.1, "fwd_ret"].mean()
                    - x["fwd_ret"].mean() if len(x) > 50 else np.nan).dropna()
                out[(LAB[s], p, tag)] = {
                    "月均只数": g.groupby("month").size().mean(),
                    "RankIC": ics.mean(), "t": tstat(ics),
                    "尾部超额(年化)": tl.mean() * 12,
                    "尾部t": stats.ttest_1samp(tl, 0)[0]}
    tab2 = pd.DataFrame(out).T.round(4)
    print("\n== 高污染行业交互 (RankIC 与尾部效应) ==")
    print(tab2.to_string())
    tab2.to_csv(config.OUT_DIR / "decomp_pollution_interaction.csv",
                encoding="utf-8-sig")
