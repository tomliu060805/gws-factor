# -*- coding: utf-8 -*-
"""月频信号面板构建管线: 行情 + ESG 信号 + Barra 中性化 (07/08 共用)."""
import numpy as np
import pandas as pd

from . import config, gws
from .marketdata import build_monthly_panel, stkcd_to_code

STYLES = ["SIZE", "BTOP", "MOM", "BETA", "RESVOL", "TURN"]
FUND = ["ESG_per", "ROA", "Lev"]


def zclip(s, k=3.0):
    z = (s - s.mean()) / s.std()
    return z.clip(-k, k)


def neutralize(d, signal, styles, with_industry=True):
    """单月截面: signal 对 风格(+行业哑变量) 回归取残差."""
    cols = [signal] + styles
    sub = d.dropna(subset=cols)
    if len(sub) < 100:
        return pd.Series(dtype=float)
    mats = [np.ones((len(sub), 1)),
            np.column_stack([zclip(sub[c]).values for c in styles])]
    if with_industry and sub["industry"].nunique() > 1:
        mats.append(pd.get_dummies(sub["industry"], drop_first=True,
                                   dtype=float).values)
    Xm = np.hstack(mats)
    y = sub[signal].values.astype(float)
    beta = np.linalg.lstsq(Xm, y, rcond=None)[0]
    return pd.Series(y - Xm @ beta, index=sub.index)


SUB_DIS = {"E": "bloomberge", "S": "bloombergs", "G": "bloombergg"}


def build_signal_panel(verbose=True, start_month=1, with_sub=False):
    """返回月频截面长表 d: ret/fwd_ret/风格因子/GWS 原始与中性化列.

    start_month: 年 t 信号从 t+1 年的第几月起应用, 覆盖其后 12 个月.
      1 (默认) = t+1 自然年 1-12 月; 7 = t+1年7月 ~ t+2年6月
      (发布时点敏感性: 假设评级最晚 t+1 年中才可得).
    """
    panel = build_monthly_panel()
    if verbose:
        print(f"   行情: {panel.month.min()} ~ {panel.month.max()}, "
              f"{panel.code.nunique()} 只股票, {len(panel):,} 行")

    wide = panel.pivot(index="month", columns="code", values="ret")
    fwd = wide.shift(-1).stack().rename("fwd_ret").reset_index()
    panel = panel.merge(fwd, on=["month", "code"], how="left")

    esg = gws.add_gws(pd.read_stata(config.PANEL_FULL))
    esg["code"] = esg["stkcd"].map(stkcd_to_code)
    keep = ["code", "year", "GWS_ind", "GWS_indyr", "ESG_per", "ESG_dis",
            "ROA", "Lev", "industry"]
    if with_sub:
        # E/S/G 分项裂口: z(分项披露|行业) − z(ESG表现|行业); 污染分组哑变量
        for k, col in SUB_DIS.items():
            esg[f"GWS{k}_ind"] = gws.zscore_gap(esg, "industry", dis=col)
            keep.append(f"GWS{k}_ind")
        for i in (1, 2, 3):
            esg[f"pollution_{i}"] = esg[f"重污染分组{i}"]
            keep.append(f"pollution_{i}")
    sig = esg.dropna(subset=["code"])[keep].copy()
    sig["sig_year"] = sig["year"].astype(int)
    yy = panel["month"].str[:4].astype(int)
    mm = panel["month"].str[5:7].astype(int)
    # 收益月 m 使用的信号年份: 应用窗口 [t+1年start_month, 其后12个月)
    panel["sig_year"] = np.where(mm >= start_month, yy - 1, yy - 2)
    panel["sig_age"] = (mm - start_month) % 12 + 1   # 信号已使用的第几个月
    d = panel.merge(sig.drop(columns="year"), on=["code", "sig_year"],
                    how="inner")
    if verbose:
        print(f"   信号: {d.month.nunique()} 个月截面, "
              f"月均 {len(d)/d.month.nunique():.0f} 只股票")

    d = d.set_index(pd.RangeIndex(len(d)))
    d["GWS_style"] = pd.concat([neutralize(g, "GWS_ind", STYLES)
                                for _, g in d.groupby("month")])
    d["GWS_full"] = pd.concat([neutralize(g, "GWS_ind", STYLES + FUND)
                               for _, g in d.groupby("month")])
    d["ESGper_style"] = pd.concat([neutralize(g, "ESG_per", STYLES)
                                   for _, g in d.groupby("month")])
    if with_sub:
        for k in SUB_DIS:
            d[f"GWS{k}_full"] = pd.concat(
                [neutralize(g, f"GWS{k}_ind", STYLES + FUND)
                 for _, g in d.groupby("month")])
    return d
