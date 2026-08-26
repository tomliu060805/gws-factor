# -*- coding: utf-8 -*-
"""2024 强制披露指引前后的沉默折价变化(探索性 DiD).

事件: 2024-04-12 沪深北交易所发布《上市公司可持续发展报告指引》, 要求
  上证180、科创50、深证100、创业板指样本股及境内外同时上市公司强制披露
  可持续发展报告(自 2025 年度报告起). 若沉默被定价源于信息壁垒, 壁垒被
  外生拆除应使受强制组的沉默折价收窄.

处理组代理: 交易所成分名单不可得, 用**事前(2024-03)市值排序**构造:
  沪市主板前180 + 深市主板前100 + 创业板前100 + 科创板前50.
  测量误差使部分处理组混入对照组, 方向上**低估**处理效应.

★★ 识别性警告 ★★
  新"国九条"(《关于加强监管防范风险推动资本市场高质量发展的若干意见》)
  于 **同日 2024-04-12** 发布, 引发大小盘剧烈分化. 本文处理组按市值构造,
  与该冲击的受益方高度重合 → 本 DiD **无法把披露强制与国九条的规模冲击
  分离**. 故本表仅作**描述性证据**, 不作为因果证据; 论文中应据此定位,
  或等 2025 年报强制披露落地后以真实披露变化重做.
  缓释手段: 报告市值分层内的匹配估计(同规模带内比较).
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
import statsmodels.api as sm
from greenwash import config
from greenwash.pipeline import build_signal_panel

SIG = "GWS_full"
EVENT = "2024-04"
PRE0, POST1 = "2023-04", "2024-12"


def board(code):
    if code.startswith("688"):
        return "STAR"
    if code.startswith("300"):
        return "ChiNext"
    return "SH" if code.endswith("XSHG") else "SZ"


def treated_set(panel, asof="2024-03"):
    """按事前市值构造强制披露代理名单."""
    g = panel[panel["month"] == asof].dropna(subset=["MCAP"]).copy()
    g["board"] = g["code"].map(board)
    caps = {"SH": 180, "SZ": 100, "ChiNext": 100, "STAR": 50}
    out = set()
    for b, n in caps.items():
        sub = g[g["board"] == b].nlargest(n, "MCAP")
        out |= set(sub["code"])
    return out


if __name__ == "__main__":
    d = build_signal_panel(verbose=False)
    trt = treated_set(d)
    print(f">> 处理组代理: {len(trt)} 只")

    dd = d.dropna(subset=[SIG, "fwd_ret", "MCAP"]).copy()
    dd = dd[(dd["month"] >= PRE0) & (dd["month"] <= POST1)]
    dd["treated"] = dd["code"].isin(trt).astype(int)
    dd["post"] = (dd["month"] >= EVENT).astype(int)
    dd["silent"] = (dd.groupby("month")[SIG].rank(pct=True) <= 0.20).astype(int)
    dd["exc"] = dd["fwd_ret"] - dd.groupby("month")["fwd_ret"].transform("mean")
    print(f"   样本 {len(dd):,} 行, {dd.month.nunique()} 个月 "
          f"(pre {int((1-dd.post).sum()):,} / post {int(dd.post.sum()):,}), "
          f"处理组占比 {dd.treated.mean():.1%}")

    # ---- 2×2×2 描述表: 沉默折价 = 沉默组超额 − 非沉默组超额 ----
    cell = dd.groupby(["treated", "post", "silent"])["exc"].mean().unstack()
    disc = (cell[1] - cell[0]).rename("沉默折价(月度)")
    tab = disc.unstack().rename(index={0: "对照组", 1: "处理组"},
                                columns={0: "事前", 1: "事后"})
    tab["DiD(事后−事前)"] = tab["事后"] - tab["事前"]
    print("\n== 2×2 描述: 月度沉默折价 (沉默组超额 − 非沉默组超额) ==")
    print((tab * 12).round(4).to_string(), "  [年化]")

    # ---- 三重差分回归 (月份固定效应, 双向聚类) ----
    X = pd.DataFrame({
        "silent": dd["silent"], "treated": dd["treated"],
        "silent_treated": dd["silent"] * dd["treated"],
        "silent_post": dd["silent"] * dd["post"],
        "treated_post": dd["treated"] * dd["post"],
        "DDD": dd["silent"] * dd["treated"] * dd["post"],
    })
    X = pd.concat([X, pd.get_dummies(dd["month"], prefix="M", drop_first=True,
                                     dtype=float)], axis=1)
    X = sm.add_constant(X)
    groups = np.column_stack([pd.factorize(dd["month"])[0],
                              pd.factorize(dd["code"])[0]])
    res = sm.OLS(dd["fwd_ret"].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    keep = ["silent", "silent_treated", "silent_post", "DDD", "treated_post"]
    ddd = pd.DataFrame({"系数(月度)": res.params[keep],
                        "年化": res.params[keep] * 12,
                        "t": res.tvalues[keep]}).round(4)
    print("\n== 三重差分 (月份FE, 月×股双向聚类) ==")
    print(ddd.to_string())

    # ---- 规模带内匹配估计(缓释国九条规模冲击) ----
    dd["sz_band"] = dd.groupby("month")["MCAP"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    rows = {}
    for b, g in dd.groupby("sz_band"):
        c = g.groupby(["treated", "post", "silent"])["exc"].mean().unstack()
        if c.shape != (4, 2):
            continue
        dsc = (c[1] - c[0]).unstack()
        rows[f"市值第{b+1}层"] = {
            "处理组DiD(年化)": (dsc.loc[1, 1] - dsc.loc[1, 0]) * 12,
            "对照组DiD(年化)": (dsc.loc[0, 1] - dsc.loc[0, 0]) * 12,
            "DDD(年化)": ((dsc.loc[1, 1] - dsc.loc[1, 0])
                        - (dsc.loc[0, 1] - dsc.loc[0, 0])) * 12,
            "处理组只数": g[g.treated == 1]["code"].nunique()}
    band = pd.DataFrame(rows).T.round(4)
    print("\n== 规模带内 DDD (缓释同期国九条规模冲击) ==")
    print(band.to_string())

    out = {"2x2": (tab * 12).round(4), "DDD": ddd, "band": band}
    for k, v in out.items():
        v.to_csv(config.OUT_DIR / f"mandate_did_{k}.csv", encoding="utf-8-sig")
    print("\n[定位] 处理组与同日国九条规模冲击的受益方重合, 本表为描述性证据, "
          "不构成因果识别; 见脚本头部说明.")
