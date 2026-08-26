# -*- coding: utf-8 -*-
"""防御性基准: 7月起应用 + 行业×年标准化 (双报告口径之防御版).

依据 (前视审计):
  - accper 证实年 t 评分为财年口径, 基于 t 年年报(t+1年4-6月发布)
    → 1月起应用乐观, 7月起应用为可辩护口径;
  - GWS_ind 行业标准化含全样本常数(跨年) → 行业×年标准化无此前视.
本脚本产出防御性配置的 IC 与全市场增强指标, 与基准口径并排双报告.
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


def enhance_metrics(d, sig):
    dd = d.dropna(subset=[sig, "fwd_ret"]).copy()
    dd["flag"] = dd.groupby("month")[sig].rank(pct=True) <= 0.10
    base = dd.groupby("month")["fwd_ret"].mean()
    enh = dd[~dd["flag"]].groupby("month")["fwd_ret"].mean()
    exc = (enh - base).dropna()
    t, _ = stats.ttest_1samp(exc, 0)
    return pd.Series({"年化超额": exc.mean() * 12,
                      "跟踪误差": exc.std() * np.sqrt(12),
                      "信息比率": exc.mean() / exc.std() * np.sqrt(12),
                      "t": t, "月胜率": (exc > 0).mean(), "月数": len(exc)})


if __name__ == "__main__":
    print(">> 基准口径 (1月起, 行业标准化) ...", flush=True)
    d1 = build_signal_panel(verbose=False)
    print(">> 防御口径 (7月起) ...", flush=True)
    d7 = build_signal_panel(verbose=False, start_month=7)
    # 防御口径信号: 行业×年标准化裂口 → 完全中性化
    d7["GWS_def"] = pd.concat([neutralize(g, "GWS_indyr", STYLES + FUND)
                               for _, g in d7.groupby("month")])

    ic_tab = pd.concat({
        "基准(1月起+行业标准化)": ic_summary(
            ic_series(d1, "GWS_full", "fwd_ret", "month", 100)),
        "防御(7月起+行业×年标准化)": ic_summary(
            ic_series(d7, "GWS_def", "fwd_ret", "month", 100)),
    }, names=["口径", "type"]).round(4)
    print("\n== IC 双报告 ==")
    print(ic_tab.to_string())
    ic_tab.to_csv(config.OUT_DIR / "defensive_ic.csv", encoding="utf-8-sig")

    enh_tab = pd.concat({
        "基准": enhance_metrics(d1, "GWS_full"),
        "防御": enhance_metrics(d7, "GWS_def"),
    }, axis=1).round(4)
    print("\n== 全市场增强(剔尾部10%, 等权月刷) 双报告 ==")
    print(enh_tab.to_string())
    enh_tab.to_csv(config.OUT_DIR / "defensive_enhance.csv",
                   encoding="utf-8-sig")
