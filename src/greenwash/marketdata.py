# -*- coding: utf-8 -*-
"""服务器行情 → 月频股票面板 (收益 + Barra 风格因子代理).

数据源 (只读, 路径见 config.py, 可用环境变量覆盖):
  <GWS_MARKET_DIR>/stock/price/price_daily/YYYY-MM-DD.parquet
      close/pre_close: 验证 pre_close 为除权除息参考价, 故
      日收益 = close/pre_close - 1 已剔除分红送转的价格跳变
  <GWS_MARKET_DIR>/stock/fundamental/valuation/  月末市值/PB/PE
  <GWS_MARKET_DIR>/index/price/price_daily/      000985.XSHG 中证全指为市场基准

Barra 风格代理(月末截面):
  SIZE   = ln(总市值)
  BTOP   = 1/PB
  MOM    = t-12..t-2 月累计收益 (12-1 动量)
  BETA   = 过去 250 交易日对全指的回归 beta (min 120d)
  RESVOL = 过去 250 交易日残差波动 (总波动扣 beta·市场 后的 std)
  TURN   = 当月成交额合计 / 月末流通市值

流动性与价格发现(月末截面, 供 H2/H3 机制检验):
  AMIHUD = 当月 mean(|日收益| / 日成交额) × 1e9  (Amihud 2002 非流动性)
  DELAY  = Hou-Moskowitz (2005) 价格延迟: 1 − R²(仅当期市场) / R²(当期+4阶滞后),
           过去 52 周周频收益估计, 越大表示价格对市场信息反应越慢

构建结果缓存至 output/monthly_panel.parquet (约数分钟, 仅首次).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PRICE_DIR = config.PRICE_DIR
VAL_DIR = config.VAL_DIR
INDEX_DIR = config.INDEX_DIR
MARKET = "000985.XSHG"
CACHE = config.OUT_DIR / "monthly_panel.parquet"

WIN_BETA, MIN_BETA = 250, 120
WIN_DELAY_W, MIN_DELAY_W = 52, 40


def _daily_files(start, end):
    return [f for f in sorted(PRICE_DIR.glob("*.parquet"))
            if start <= f.stem <= end]


def build_daily_returns(start="2014-01-01", end="2024-12-31"):
    """宽表日收益 (index=date, columns=code) 与日成交额."""
    rets, moneys = {}, {}
    for f in _daily_files(start, end):
        d = pd.read_parquet(f, columns=["code", "close", "pre_close",
                                        "paused", "money"])
        d = d[~d["paused"] & (d["pre_close"] > 0)]
        rets[f.stem] = pd.Series((d["close"] / d["pre_close"] - 1).values,
                                 index=d["code"].values)
        moneys[f.stem] = pd.Series(d["money"].values, index=d["code"].values)
    ret = pd.DataFrame(rets).T.astype("float32")
    money = pd.DataFrame(moneys).T.astype("float64")
    ret.index = money.index = pd.to_datetime(ret.index)
    return ret, money


def market_returns(dates):
    out = {}
    for dt in dates:
        f = INDEX_DIR / f"{dt.date()}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["code", "close", "pre_close"])
        row = d[d["code"] == MARKET]
        if len(row):
            out[dt] = float(row["close"].iloc[0] / row["pre_close"].iloc[0] - 1)
    return pd.Series(out, name="mkt")


def rolling_beta_resvol(ret, mkt, month_ends):
    """每个月末: 过去 WIN_BETA 日的 beta 与残差波动 (矩阵化逐月末计算)."""
    betas, resvols = {}, {}
    for me in month_ends:
        win = ret.loc[:me].tail(WIN_BETA)
        m = mkt.reindex(win.index)
        ok = m.notna()
        win, m = win[ok], m[ok].values
        n = win.notna().sum()
        mm = m - m.mean()
        r = win.fillna(0.0).values
        valid = win.notna().values
        # 逐列 cov/var, 缺失日不计入
        m_mat = np.where(valid, mm[:, None], 0.0)
        cov = (r * m_mat).sum(0) / np.maximum(valid.sum(0) - 1, 1)
        var_m = (m_mat**2).sum(0) / np.maximum(valid.sum(0) - 1, 1)
        beta = np.where(var_m > 0, cov / np.where(var_m > 0, var_m, 1), np.nan)
        resid = np.where(valid, r - beta[None, :] * np.where(valid, m[:, None], 0), np.nan)
        resvol = pd.DataFrame(resid, columns=win.columns).std().values
        bad = n.values < MIN_BETA
        beta[bad] = np.nan
        resvol[bad] = np.nan
        betas[me] = pd.Series(beta, index=win.columns)
        resvols[me] = pd.Series(resvol, index=win.columns)
    return pd.DataFrame(betas).T, pd.DataFrame(resvols).T


def price_delay(ret, mkt, month_ends):
    """Hou-Moskowitz (2005) 价格延迟, 每个月末用过去 52 周周频收益估计.

    DELAY = 1 − R²(仅当期市场) / R²(当期 + 4 阶滞后市场)
    回归元(市场收益及其滞后)对所有股票相同, 故一次性解出全截面.
    停牌缺失周填 0, 要求窗口内有效周数 ≥ MIN_DELAY_W.
    """
    wk = ret.groupby(ret.index.to_period("W")).apply(
        lambda g: (1 + g).prod(min_count=1) - 1)
    wm = mkt.groupby(mkt.index.to_period("W")).apply(lambda g: (1 + g).prod() - 1)
    lags = pd.concat([wm.shift(k).rename(k) for k in range(5)], axis=1)

    def r2(Y, X):
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        ssr = ((Y - X @ beta) ** 2).sum(0)
        sst = ((Y - Y.mean(0)) ** 2).sum(0)
        return 1 - ssr / np.where(sst > 0, sst, np.nan)

    out = {}
    for me in month_ends:
        w = wk.loc[:me.to_period("W")].tail(WIN_DELAY_W)
        L = lags.reindex(w.index)
        if len(w) < WIN_DELAY_W or L.isna().any().any():
            continue
        valid = w.notna().sum(0).values
        Y = w.fillna(0.0).values
        one = np.ones((len(w), 1))
        r2_1 = r2(Y, np.hstack([one, L[[0]].values]))
        r2_5 = r2(Y, np.hstack([one, L.values]))
        d = 1 - r2_1 / np.where(r2_5 > 0.01, r2_5, np.nan)
        d = np.where((valid >= MIN_DELAY_W) & np.isfinite(d), d, np.nan)
        out[me] = pd.Series(np.clip(d, 0, 1), index=w.columns)
    return pd.DataFrame(out).T


def build_monthly_panel(force=False):
    """长表: month, code, ret(当月), SIZE/BTOP/MOM/BETA/RESVOL/TURN/AMIHUD/DELAY."""
    need = {"ret", "SIZE", "BTOP", "EP", "MCAP", "MOM", "BETA", "RESVOL",
            "TURN", "AMIHUD", "DELAY"}
    if CACHE.exists() and not force:
        cached = pd.read_parquet(CACHE)
        if need <= set(cached.columns):
            return cached
        print("   [缓存缺少流动性/延迟列, 重建月频面板 ...]", flush=True)

    ret, money = build_daily_returns()
    mkt = market_returns(ret.index)

    # Amihud 非流动性 (日 |收益|/成交额 的月均, 十亿元口径)
    illiq_d = ret.abs() / money.where(money > 0) * 1e9
    amihud = illiq_d.groupby(illiq_d.index.to_period("M")).mean()

    grp = ret.groupby(ret.index.to_period("M"))
    mret = grp.apply(lambda g: (1 + g).prod(min_count=5) - 1).astype("float32")
    mdays = grp.count()
    mret = mret.where(mdays >= 5)
    mmoney = money.groupby(money.index.to_period("M")).sum(min_count=5)
    month_ends = [g.index.max() for _, g in grp]

    beta, resvol = rolling_beta_resvol(ret, mkt, month_ends)
    beta.index = resvol.index = mret.index
    delay = price_delay(ret, mkt, month_ends)
    delay.index = [d.to_period("M") for d in delay.index]

    # 月末估值
    vals = {}
    for me in month_ends:
        f = VAL_DIR / f"{me.date()}.parquet"
        if f.exists():
            v = pd.read_parquet(f, columns=["code", "market_cap", "pb_ratio",
                                            "pe_ratio", "circulating_market_cap"])
            vals[me.to_period("M")] = v.set_index("code")
    rows = []
    for m in mret.index:
        if m not in vals:
            continue
        v = vals[m]
        df = pd.DataFrame({
            "ret": mret.loc[m],
            "SIZE": np.log(v["market_cap"].where(v["market_cap"] > 0) * 1e8),
            "BTOP": 1.0 / v["pb_ratio"].where(v["pb_ratio"] > 0),
            "EP": 1.0 / v["pe_ratio"].where(v["pe_ratio"] > 0),
            "MCAP": v["market_cap"] * 1e8,
            "TURN": mmoney.loc[m] / (v["circulating_market_cap"] * 1e8),
            "BETA": beta.loc[m],
            "RESVOL": resvol.loc[m],
            "AMIHUD": amihud.loc[m] if m in amihud.index else np.nan,
            "DELAY": delay.loc[m] if m in delay.index else np.nan,
        })
        df["month"] = str(m)
        rows.append(df.rename_axis("code").reset_index())
    panel = pd.concat(rows, ignore_index=True).dropna(subset=["ret"])

    # 12-1 动量
    wide = panel.pivot(index="month", columns="code", values="ret")
    log1p = np.log1p(wide)
    mom = np.expm1(log1p.rolling(11).sum().shift(2))
    panel = panel.merge(mom.stack().rename("MOM").reset_index(),
                        on=["month", "code"], how="left")
    panel.to_parquet(CACHE)
    return panel


def stkcd_to_code(stkcd):
    s = f"{int(stkcd):06d}"
    if s[0] == "6":
        return s + ".XSHG"
    if s[0] in ("0", "3"):
        return s + ".XSHE"
    return None
