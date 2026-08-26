# -*- coding: utf-8 -*-
"""月频因子检验 + Barra 风格中性化 —— 年频(06)的高功效升级版.

信号: 第 t 年 GWS → 应用于 t+1 自然年的 12 个月末截面 (与 06 的年频对齐一致)
收益: 服务器日行情复权日收益复合成月收益, 前瞻 1 个月
中性化: 逐月截面残差化于 Barra 风格代理
  [SIZE, BTOP, MOM, BETA, RESVOL, TURN] + 行业哑变量 + [ESG_per, ROA, Lev]
输出: 原始/仅风格中性/风格+ESG水平中性 三个层次的 IC/ICIR/RankIC/RankICIR,
      十分组月频多空(含年化/胜率/盈亏比/盈利因子).

运行环境需含 pyarrow (如 conda env rdagent); 首次运行构建月频行情缓存约数分钟.
"""
import _path  # noqa: F401
import sys

try:
    import pyarrow  # noqa: F401
except ImportError:
    sys.exit("需要 pyarrow 读取行情数据: pip install pyarrow "
             "(见 README 的环境说明)")

import numpy as np
import pandas as pd
from scipy import stats
from greenwash import config
from greenwash.factor import ic_series, ic_summary
from greenwash.pipeline import build_signal_panel


def decile_ls(d, signal, fwd="fwd_ret", n=10):
    dd = d.dropna(subset=[signal, fwd]).copy()
    dd["grp"] = dd.groupby("month")[signal].transform(
        lambda s: pd.qcut(s.rank(method="first"), n, labels=False) + 1)
    yearly = dd.pivot_table(index="month", columns="grp", values=fwd)
    ls = (yearly[n] - yearly[1]).dropna()
    t, p = stats.ttest_1samp(ls, 0)
    return yearly.mean(), pd.Series({
        "月均多空": ls.mean(), "年化": ls.mean() * 12,
        "年化波动": ls.std() * np.sqrt(12),
        "Sharpe": ls.mean() / ls.std() * np.sqrt(12), "t": t, "p": p,
        "月度胜率": (ls > 0).mean(), "触发月数": len(ls),
        "盈亏比": ls[ls > 0].mean() / abs(ls[ls < 0].mean()),
        "盈利因子": ls[ls > 0].sum() / abs(ls[ls < 0].sum()),
    })


if __name__ == "__main__":
    print(">> 构建/加载月频信号面板 ...", flush=True)
    d = build_signal_panel()

    summ = {}
    for s, lab in [("GWS_ind", "GWS原始"), ("GWS_style", "GWS+风格中性"),
                   ("GWS_full", "GWS+风格+ESG/基本面中性"),
                   ("ESG_per", "ESG_per原始"), ("ESGper_style", "ESG_per+风格中性")]:
        summ[lab] = ic_summary(ic_series(d, s, "fwd_ret", time="month",
                                         min_n=100))
    tab = pd.concat(summ, names=["signal", "type"]).round(4)
    print("\n== 月频 IC 汇总 (IC/ICIR/RankIC/RankICIR) ==")
    print(tab.to_string())
    tab.to_csv(config.OUT_DIR / "factor_ic_monthly_barra.csv",
               encoding="utf-8-sig")

    for s in ("GWS_ind", "GWS_full"):
        grp_mean, ls = decile_ls(d, s)
        print(f"\n== {s} 十分组月均前瞻收益 ==")
        print(grp_mean.round(4).to_string())
        print(f"-- 多空(G10-G1) --\n{ls.round(4).to_string()}")
        ls.round(6).to_csv(config.OUT_DIR / f"factor_monthly_ls_{s}.csv",
                           encoding="utf-8-sig")
