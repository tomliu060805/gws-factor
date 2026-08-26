# -*- coding: utf-8 -*-
"""信息壁垒机制检验 H2/H3: 流动性渠道与价格发现速度.

理论链条 (Merton 1987 认知假说 / Easley-O'Hara 2004 信息风险定价):
  披露不足(壁垒) → 信息环境恶化(流动性差 H2、价格发现慢 H3) → 截面折价

H2 流动性渠道:
  a) 低披露组的 Amihud 非流动性是否更高 (十分组 + Fama-MacBeth)
  b) 因子存活性: GWS 追加 ln(AMIHUD) 中性化后 IC 是否存活
     (存活 → 信息风险定价; 被吸收 → 只是流动性溢价的马甲)
  c) 流动性中介: GWS_t → ln(AMIHUD)_t → ret_{t+1} 的 FM 三步法 + Sobel
H3 价格发现:
  a) 低披露组的 Hou-Moskowitz 价格延迟 DELAY 是否更高
  b) 双重排序: 沉默溢价是否集中在高延迟(壁垒高)子样本

注: GWS_full 已对 SIZE/BTOP/MOM/BETA/RESVOL/TURN + 行业 + 基本面中性化,
    故其与 AMIHUD/DELAY 的关系不是市值或换手的机械映射.
运行: 需 pyarrow (见 README); 首次运行会重建月频面板(含DELAY)约数分钟.
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

SIG = "GWS_full"


def fm(d, y, xs, min_n=100):
    """Fama-MacBeth: 逐月截面 OLS, 返回系数均值与 t."""
    rows = []
    for _, g in d.groupby("month"):
        sub = g[[y] + xs].dropna()
        if len(sub) < min_n:
            continue
        X = np.column_stack([np.ones(len(sub)), sub[xs].to_numpy(float)])
        rows.append(np.linalg.lstsq(X, sub[y].to_numpy(float), rcond=None)[0])
    b = pd.DataFrame(rows, columns=["const"] + xs)
    T = len(b)
    return pd.DataFrame({"系数": b.mean(),
                         "t": b.mean() / b.std() * np.sqrt(T)}), b, T


def tstat(s):
    return s.mean() / s.std() * np.sqrt(len(s))


if __name__ == "__main__":
    print(">> 构建信号面板 (含 AMIHUD/DELAY) ...", flush=True)
    d = build_signal_panel(verbose=False)
    d["lnAMIHUD"] = np.log(d["AMIHUD"].where(d["AMIHUD"] > 0))
    cov = d[["AMIHUD", "DELAY"]].notna().mean()
    print(f"   覆盖: AMIHUD {cov['AMIHUD']:.1%}, DELAY {cov['DELAY']:.1%}, "
          f"{d.month.nunique()} 个月")

    # ================= H2 (a) 分组与 FM =================
    dd = d.dropna(subset=[SIG, "lnAMIHUD"]).copy()
    dd["grp"] = dd.groupby("month")[SIG].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    g_tab = dd.groupby("grp").agg(
        Amihud均值=("AMIHUD", "mean"), lnAmihud=("lnAMIHUD", "mean"),
        DELAY均值=("DELAY", "mean"), 只数=("code", "size"))
    print("\n== H2a. GWS 十分组的信息环境 (G1=最沉默) ==")
    print(g_tab.round(4).to_string())
    g_tab.round(4).to_csv(config.OUT_DIR / "barrier_decile_env.csv",
                          encoding="utf-8-sig")

    # 逐月截面: 沉默是否预测更差流动性 (GWS_full 已剥离规模/换手)
    res_a, _, T = fm(d, "lnAMIHUD", [SIG])
    res_a2, _, _ = fm(d, "lnAMIHUD", [SIG, "SIZE", "TURN"])
    print(f"\n== H2a. FM: ln(Amihud) ~ GWS_full  (T={T}) ==")
    print("单变量:\n", res_a.round(4).to_string())
    print("加控制:\n", res_a2.round(4).to_string())

    # ================= H2 (b) 因子存活性 =================
    d["GWS_liq"] = pd.concat(
        [neutralize(g, "GWS_ind", STYLES + FUND + ["lnAMIHUD"])
         for _, g in d.groupby("month")])
    d["GWS_liqdelay"] = pd.concat(
        [neutralize(g, "GWS_ind", STYLES + FUND + ["lnAMIHUD", "DELAY"])
         for _, g in d.groupby("month")])
    surv = pd.concat({
        "基准 GWS_full": ic_summary(ic_series(d, SIG, "fwd_ret", "month", 100)),
        "+Amihud 中性化": ic_summary(ic_series(d, "GWS_liq", "fwd_ret", "month", 100)),
        "+Amihud&Delay 中性化": ic_summary(
            ic_series(d, "GWS_liqdelay", "fwd_ret", "month", 100)),
    }, names=["口径", "type"]).round(4)
    print("\n== H2b. 因子存活性 (追加流动性/延迟中性化) ==")
    print(surv.to_string())
    surv.to_csv(config.OUT_DIR / "barrier_survival.csv", encoding="utf-8-sig")

    # ================= H2 (c) 流动性中介 =================
    a_res, a_b, _ = fm(d, "lnAMIHUD", [SIG])
    b_res, b_b, Tm = fm(d, "fwd_ret", ["lnAMIHUD", SIG])
    a, sa = a_b[SIG].mean(), a_b[SIG].std() / np.sqrt(len(a_b))
    b, sb = b_b["lnAMIHUD"].mean(), b_b["lnAMIHUD"].std() / np.sqrt(len(b_b))
    c_res, c_b, _ = fm(d, "fwd_ret", [SIG])
    z = a * b / np.sqrt(b**2 * sa**2 + a**2 * sb**2)
    med = pd.Series({
        "a: GWS→lnAmihud": a, "a_t": a / sa,
        "b: lnAmihud→ret(控GWS)": b, "b_t": b / sb,
        "c: 总效应 GWS→ret": c_b[SIG].mean(), "c_t": tstat(c_b[SIG]),
        "c': 直接效应(控Amihud)": b_b[SIG].mean(), "c'_t": tstat(b_b[SIG]),
        "间接效应 a*b": a * b, "Sobel_z": z,
        "Sobel_p": 2 * (1 - stats.norm.cdf(abs(z))),
        "中介占比": a * b / c_b[SIG].mean() if c_b[SIG].mean() else np.nan,
    })
    print("\n== H2c. 流动性中介 (Fama-MacBeth 三步法) ==")
    print(med.round(5).to_string())
    med.round(6).to_csv(config.OUT_DIR / "barrier_mediation.csv",
                        encoding="utf-8-sig")

    # ================= H3 价格发现 =================
    res_d, _, Td = fm(d, "DELAY", [SIG])
    res_d2, _, _ = fm(d, "DELAY", [SIG, "SIZE", "TURN"])
    print(f"\n== H3a. FM: 价格延迟 DELAY ~ GWS_full  (T={Td}) ==")
    print("单变量:\n", res_d.round(4).to_string())
    print("加控制:\n", res_d2.round(4).to_string())

    def double_sort(dd, var, label):
        """按 var 月度中位数二分, 各子样本内的沉默溢价强度."""
        dd = dd.dropna(subset=[SIG, "fwd_ret", var]).copy()
        med = dd.groupby("month")[var].transform("median")
        dd["g"] = np.where(dd[var] > med, "高", "低")
        rows = {}
        for tag, g in dd.groupby("g"):
            ics = ic_series(g, SIG, "fwd_ret", "month", 50)["RankIC"]
            tl = g.groupby("month").apply(
                lambda x: x.loc[x[SIG].rank(pct=True) <= 0.1, "fwd_ret"].mean()
                - x["fwd_ret"].mean()).dropna()
            rows[f"{label}-{tag}"] = {
                "月均只数": g.groupby("month").size().mean(),
                "RankIC": ics.mean(), "t": tstat(ics),
                "尾部超额(年化)": tl.mean() * 12,
                "尾部t": stats.ttest_1samp(tl, 0)[0]}
        return pd.DataFrame(rows).T

    tab_d = pd.concat([double_sort(d, "DELAY", "价格延迟"),
                       double_sort(d, "RESVOL", "残差波动"),
                       double_sort(d, "lnAMIHUD", "非流动性"),
                       double_sort(d, "SIZE", "市值")]).round(4)
    print("\n== H3b. 双重排序: 沉默溢价在什么环境里最强 ==")
    print(tab_d.to_string())
    tab_d.to_csv(config.OUT_DIR / "barrier_delay_doublesort.csv",
                 encoding="utf-8-sig")

    # 判别检验: 价格发现慢 vs 套利受限(残差波动) —— 在低波动子样本内再看延迟
    low_vol = d[d["RESVOL"] <= d.groupby("month")["RESVOL"].transform("median")]
    tab_t = double_sort(low_vol, "DELAY", "低波动内-延迟").round(4)
    print("\n== H3c. 判别检验: 低残差波动子样本内, 延迟仍有判别力? ==")
    print(tab_t.to_string())
    tab_t.to_csv(config.OUT_DIR / "barrier_delay_triple.csv",
                 encoding="utf-8-sig")
