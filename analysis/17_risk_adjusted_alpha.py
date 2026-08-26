# -*- coding: utf-8 -*-
"""风险调整后 alpha: 组合收益对 CAPM / FF3 / 中国三因子(CH-3)回归.

被检验组合(均为月调仓、等权、无成本; 收益记在持有月):
  1) 多空 G10−G1
  2) 沉默组 G1 相对全样本(负向 alpha 是折价的直接体现)
  3) 全市场增强: 剔除尾部10%后的组合 相对 覆盖域等权基准
因子: 自建 CH-3 (Liu-Stambaugh-Yuan 2019: 剔小30% + EP 价值口径) 与
      FF3 对照口径(不剔小票, B/M 价值), 无风险利率用 CSMAR 月度化利率.
标准误: Newey-West (6 期).

【定位】本表检验的是**可预测性经风险因子调整后是否仍存在**, 不构成因果推断.
运行: 见 README 环境说明
"""
import _path  # noqa: F401
import sys

try:
    import pyarrow  # noqa: F401
except ImportError:
    sys.exit("需要 pyarrow: pip install pyarrow")

import numpy as np
import pandas as pd
from greenwash import config
from greenwash.chfactors import build_factors, alpha_table
from greenwash.pipeline import build_signal_panel

SIG = "GWS_full"


def shift_to_holding(s):
    """信号月 → 持有月(次月)."""
    s = s.copy()
    s.index = (pd.PeriodIndex(s.index, freq="M") + 1).astype(str)
    return s


if __name__ == "__main__":
    print(">> 构建因子与组合 ...", flush=True)
    f_ch = build_factors(ch3=True)
    f_ff = build_factors(ch3=False)
    print(f"   CH-3: {len(f_ch)} 个月 {f_ch.index.min()}~{f_ch.index.max()}")
    ann = lambda s: (s.mean() * 12, s.std() * np.sqrt(12))
    print("   因子年化(均值,波动): " + ", ".join(
        f"{c}={ann(f_ch[c])[0]:+.2%}/{ann(f_ch[c])[1]:.1%}"
        for c in ("MKT", "SMB", "VMG")))

    def make_ports(d, sig, tag):
        dd = d.dropna(subset=[sig, "fwd_ret"]).copy()
        dd["grp"] = dd.groupby("month")[sig].transform(
            lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
        gm = dd.pivot_table(index="month", columns="grp", values="fwd_ret")
        uni = dd.groupby("month")["fwd_ret"].mean()
        enh = dd[dd["grp"] > 1].groupby("month")["fwd_ret"].mean()
        return {
            f"{tag}多空 G10−G1": shift_to_holding(gm[10] - gm[1]),
            f"{tag}沉默组 G1 (超额)": shift_to_holding(gm[1] - uni),
            f"{tag}增强组合 (剔尾部10%)": shift_to_holding(enh - uni),
            f"{tag}沉默组 G1 (原始收益)": shift_to_holding(gm[1]),
            f"{tag}多头组 G10 (原始收益)": shift_to_holding(gm[10]),
        }

    d = build_signal_panel(verbose=False)
    ports = make_ports(d, SIG, "")
    # 防御口径(7月起应用 + 行业×年标准化), 与论文主口径一致
    from greenwash.pipeline import neutralize, STYLES, FUND
    d7 = build_signal_panel(verbose=False, start_month=7)
    d7["GWS_def"] = pd.concat([neutralize(g, "GWS_indyr", STYLES + FUND)
                               for _, g in d7.groupby("month")])
    ports.update(make_ports(d7, "GWS_def", "[防御]"))
    # 前三个是对冲/超额组合(不减 rf), 后两个是原始持仓(减 rf)
    is_excess = {k: k.endswith("(原始收益)") for k in ports}

    out = {}
    for name, r in ports.items():
        for lab, fac in (("CH3", f_ch), ("FF3", f_ff)):
            t = alpha_table(r, fac, excess=is_excess[name])
            for spec in t.index:
                if lab == "FF3" and spec == "CAPM":
                    continue                      # CAPM 只报一次
                out[(name, f"{lab}-{spec}" if spec != "CAPM" else "CAPM")] = {
                    "alpha(年化)": t.loc[spec, "alpha(年化)"],
                    "alpha_t": t.loc[spec, "alpha_t"],
                    "beta_MKT": t.loc[spec, "beta_MKT"],
                    "beta_SMB": t.loc[spec].get("beta_SMB", np.nan),
                    "beta_价值": t.loc[spec].get(
                        "beta_VMG", t.loc[spec].get("beta_HML", np.nan)),
                    "R2": t.loc[spec, "R2"], "月数": t.loc[spec, "月数"]}
    tab = pd.DataFrame(out).T.round(4)
    print("\n== 风险调整后 alpha (Newey-West 6期) ==")
    print(tab.to_string())
    tab.to_csv(config.OUT_DIR / "alpha_risk_adjusted.csv", encoding="utf-8-sig")
    f_ch.round(6).to_csv(config.OUT_DIR / "ch3_factors.csv", encoding="utf-8-sig")
