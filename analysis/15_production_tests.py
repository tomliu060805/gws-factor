# -*- coding: utf-8 -*-
"""入库前生产化检验: 质量基底边际增量 / 成本 / 市值加权容量 / 滚动健康门槛.

③ 质量基底: Q = z(ROA) − z(Lev) − z(RESVOL) 月度截面;
   基底组合 = Q 前30%等权; 检验在其上叠加 GWS 负面清单的边际贡献,
   以及 Fama-MacBeth 中 GWS 对 Q 的边际显著性.
④ 成本: 等权月刷增强的逐月单边换手 × 成本率(20bp/30bp 往返两档),
   报告净超额.
⑤ 市值加权: 基准与增强均按市值加权 → 容量口径的超额.
⑥ 健康门槛: 24个月滚动 RankIC t<0 → 次月停用清单; 报告门槛版表现.
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
from greenwash.factor import ic_series
from greenwash.pipeline import build_signal_panel


def z(g):
    return (g - g.mean()) / g.std()


def ann(exc):
    t, _ = stats.ttest_1samp(exc, 0)
    return {"年化超额": exc.mean() * 12, "跟踪误差": exc.std() * np.sqrt(12),
            "信息比率": exc.mean() / exc.std() * np.sqrt(12), "t": t,
            "月胜率": (exc > 0).mean(), "月数": len(exc)}


def weights(dd, keep_mask, cap=None):
    """月度权重长表: 等权或市值加权."""
    k = dd[keep_mask].copy()
    if cap is None:
        k["w"] = 1.0
    else:
        k["w"] = np.exp(k["SIZE"])          # SIZE = ln(总市值)
    k["w"] = k["w"] / k.groupby("month")["w"].transform("sum")
    return k


def port_ret(k):
    return (k["w"] * k["fwd_ret"]).groupby(k["month"]).sum()


def turnover(k):
    """单边换手: 0.5*Σ|w_t − w_{t-1}| (按月)."""
    piv = k.pivot_table(index="month", columns="code", values="w",
                        fill_value=0.0)
    return (piv.diff().abs().sum(axis=1) * 0.5).iloc[1:]


if __name__ == "__main__":
    d = build_signal_panel(verbose=False)
    dd = d.dropna(subset=["GWS_full", "fwd_ret", "SIZE"]).copy()
    dd["flag"] = dd.groupby("month")["GWS_full"].rank(pct=True) <= 0.10

    # ---- ③ 质量基底边际增量 ----
    for c in ("ROA", "Lev", "RESVOL"):
        dd[f"z{c}"] = dd.groupby("month")[c].transform(z)
    dd["Q"] = dd["zROA"] - dd["zLev"] - dd["zRESVOL"]
    dd["zGWS"] = dd.groupby("month")["GWS_full"].transform(z)
    print("corr(Q, GWS_full) =",
          round(dd[["Q", "zGWS"]].dropna().Q.corr(dd.zGWS), 3))
    top = dd["Q"] >= dd.groupby("month")["Q"].transform(
        lambda s: s.quantile(0.7))
    qbase = port_ret(weights(dd, top))
    qenh = port_ret(weights(dd, top & ~dd["flag"]))
    cut_n = dd[top & dd["flag"]].groupby("month").size().mean()
    diff = (qenh - qbase).dropna()
    uni = dd.groupby("month")["fwd_ret"].mean()
    tab3 = pd.DataFrame({
        "质量基底(Q前30%)": ann((qbase - uni).dropna()),
        "基底+GWS清单": ann((qenh - uni).dropna()),
        "边际(清单贡献)": ann(diff)}).round(4)
    tab3.loc["月均剔除只数"] = ["-", "-", round(cut_n, 1)]
    print("\n== ③ 质量基底上的边际增量 (相对全样本等权) ==")
    print(tab3.to_string())
    tab3.to_csv(config.OUT_DIR / "prod_quality_marginal.csv",
                encoding="utf-8-sig")
    # Fama-MacBeth 边际显著性
    fm = []
    for m, g in dd.dropna(subset=["Q", "zGWS"]).groupby("month"):
        Xm = np.column_stack([np.ones(len(g)), z(g["Q"]), g["zGWS"]])
        fm.append(np.linalg.lstsq(Xm, g["fwd_ret"].values, rcond=None)[0])
    fm = pd.DataFrame(fm, columns=["const", "Q", "GWS"])
    print("FM: Q t={:.2f}, GWS边际 t={:.2f}".format(
        *(fm[c].mean() / fm[c].std() * np.sqrt(len(fm)) for c in ("Q", "GWS"))))

    # ---- ④ 成本版 (等权月刷) ----
    base_k, enh_k = weights(dd, dd.index == dd.index), weights(dd, ~dd["flag"])
    exc = (port_ret(enh_k) - port_ret(base_k)).dropna()
    incr_to = (turnover(enh_k) - turnover(base_k)).reindex(exc.index).fillna(0)
    rows = {"无成本": ann(exc)}
    for bp in (20, 30):
        net = exc - incr_to * bp / 1e4
        rows[f"往返{bp}bp"] = ann(net)
    tab4 = pd.DataFrame(rows).round(4)
    tab4.loc["月均增量单边换手"] = round(incr_to.mean(), 4)
    print("\n== ④ 成本版: 等权月刷增强 (增量换手计费) ==")
    print(tab4.to_string())
    tab4.to_csv(config.OUT_DIR / "prod_cost.csv", encoding="utf-8-sig")

    # ---- ⑤ 市值加权版 ----
    excw = (port_ret(weights(dd, ~dd["flag"], cap=True))
            - port_ret(weights(dd, dd.index == dd.index, cap=True))).dropna()
    tab5 = pd.DataFrame({"等权": ann(exc), "市值加权": ann(excw)}).round(4)
    print("\n== ⑤ 市值加权容量口径 ==")
    print(tab5.to_string())
    tab5.to_csv(config.OUT_DIR / "prod_capweight.csv", encoding="utf-8-sig")

    # ---- ⑥ 滚动健康门槛 ----
    ics = ic_series(d, "GWS_full", "fwd_ret", "month", 100)["RankIC"]
    roll_t = ics.rolling(24).apply(
        lambda s: s.mean() / s.std() * np.sqrt(len(s)))
    gate = (roll_t.shift(1) > 0)                    # 用截至上月的信息
    exc_g = exc.copy()
    aligned_gate = gate.reindex(exc.index)
    live = aligned_gate.notna()
    exc_g[live & ~aligned_gate.fillna(True)] = 0.0  # 门槛关闭→不挂清单
    tab6 = pd.DataFrame({"无门槛": ann(exc), "24m t>0 门槛": ann(exc_g)}).round(4)
    tab6.loc["门槛关闭月数"] = ["-", int((live & ~aligned_gate.fillna(True)).sum())]
    sub = exc.index >= "2023-01"
    tab6.loc["2023起年化超额"] = [round(exc[sub].mean() * 12, 4),
                             round(exc_g[sub].mean() * 12, 4)]
    print("\n== ⑥ 滚动健康门槛 (24个月RankIC t<0 → 停用清单) ==")
    print(tab6.to_string())
    tab6.to_csv(config.OUT_DIR / "prod_health_gate.csv", encoding="utf-8-sig")
