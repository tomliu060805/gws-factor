# -*- coding: utf-8 -*-
"""漂绿测度深化 III: 硬行为锚定旗标 + 处罚曝光事件研究.

数据: 巨潮公告 2014-2024 标题含"处罚"的公司公告 (crawler/fetch_cninfo_penalty.py)
  分类: 环保类(环保/环境/生态/排污/应急管理) / 证券监管类(证监/交易所/立案) / 其他

(4) 硬行为旗标: "披露说得漂亮(GWS高) × 近12个月吃过环保处罚" = 实锤漂绿.
    2×2 分组(是否近期环保处罚 × GWS_full 高低)比较次月收益.
(5) 曝光事件研究: 处罚公告日 [-5,+20] 市场调整累计超额收益(基准000985),
    按公告前的 GWS_full 高低分组 —— 检验"漂绿平时不跌, 曝光才跌".
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
from greenwash.marketdata import build_daily_returns, market_returns, stkcd_to_code
from greenwash.pipeline import build_signal_panel

PEN = config.DATA_EXT / "cninfo_penalty_2014_2024.parquet"
ENV_RE = "环保|环境|生态|排污|大气|水污染|应急管理"
SEC_RE = "证监|交易所|立案|警示"


def load_events():
    p = pd.read_parquet(PEN)
    p["date"] = pd.to_datetime(p["公告时间"], format="mixed").dt.normalize()
    p["month"] = p["date"].dt.strftime("%Y-%m")
    t = p["公告标题"].astype(str)
    p["cat"] = np.where(t.str.contains(ENV_RE), "环保",
                        np.where(t.str.contains(SEC_RE), "证券监管", "其他"))
    p["code"] = p["symbol"].map(lambda s: stkcd_to_code(int(s)))
    return p.dropna(subset=["code"])


def tstat(s):
    return s.mean() / s.std() * np.sqrt(len(s))


if __name__ == "__main__":
    ev = load_events()
    print(f"处罚公告 {len(ev)} 条, 涉及 {ev.code.nunique()} 只股票")
    print(ev["cat"].value_counts().to_string())

    d = build_signal_panel(verbose=False)

    # ---- (4) 硬行为旗标: 近12个月环保处罚 × GWS 高低 ----
    env = ev[ev["cat"] == "环保"]
    pen_m = set(map(tuple, env[["code", "month"]].values))
    months = sorted(d["month"].unique())
    idx = {m: i for i, m in enumerate(months)}
    pen_recent = set()
    for c, m in pen_m:
        if m[:4] < "2014" or m not in idx:
            continue
        for j in range(idx[m], min(idx[m] + 12, len(months))):
            pen_recent.add((c, months[j]))
    dd = d.dropna(subset=["GWS_full", "fwd_ret"]).copy()
    dd["pen12"] = [((c, m) in pen_recent)
                   for c, m in zip(dd["code"], dd["month"])]
    med = dd.groupby("month")["GWS_full"].transform("median")
    dd["hi_gws"] = dd["GWS_full"] > med
    uni = dd.groupby("month")["fwd_ret"].transform("mean")
    dd["exc"] = dd["fwd_ret"] - uni
    rows = {}
    for pen in (True, False):
        for hi in (True, False):
            g = dd[(dd["pen12"] == pen) & (dd["hi_gws"] == hi)]
            gm = g.groupby("month")["exc"].mean().dropna()
            rows[(f"环保处罚12m={'有' if pen else '无'}",
                  f"GWS{'高' if hi else '低'}")] = {
                "月均只数": g.groupby("month").size().mean(),
                "月度数": len(gm), "超额(年化)": gm.mean() * 12,
                "t": tstat(gm)}
    tab = pd.DataFrame(rows).T.round(4)
    print("\n== (4) 硬行为旗标 2×2: 近12个月环保处罚 × GWS_full 高低 ==")
    print(tab.to_string())
    tab.to_csv(config.OUT_DIR / "penalty_flag_2x2.csv", encoding="utf-8-sig")

    # ---- (5) 曝光事件研究 ----
    print("\n>> 加载日行情做事件研究 ...", flush=True)
    ret, _ = build_daily_returns()
    mkt = market_returns(ret.index)
    ar = ret.sub(mkt, axis=0)                       # 市场调整超额
    dates = ret.index
    sig_lookup = d.set_index(["code", "month"])["GWS_full"]

    def car(code, dt0, a=-5, b=20):
        if code not in ar.columns:
            return None
        pos = dates.searchsorted(dt0)
        if pos + a < 0 or pos + b >= len(dates):
            return None
        w = ar[code].iloc[pos + a: pos + b + 1]
        return w.values if w.notna().sum() > (b - a) * 0.7 else None

    res = {"高漂绿(GWS高)": [], "低漂绿(GWS低)": []}
    used = 0
    for _, r in ev.iterrows():   # 全部处罚类别一起; 事件月的前一月信号
        pm = (pd.Period(r["month"], "M") - 1).strftime("%Y-%m")
        s = sig_lookup.get((r["code"], pm), np.nan)
        if np.isnan(s):
            continue
        m_med = d[d["month"] == pm]["GWS_full"].median()
        c = car(r["code"], r["date"])
        if c is None:
            continue
        res["高漂绿(GWS高)" if s > m_med else "低漂绿(GWS低)"].append(c)
        used += 1
    print(f"可用事件 {used} 条 (高漂绿 {len(res['高漂绿(GWS高)'])}, "
          f"低漂绿 {len(res['低漂绿(GWS低)'])})")

    out = {}
    for k, arr in res.items():
        if not arr:
            continue
        m = np.vstack(arr)
        m = np.where(np.isnan(m), 0, m)
        cars = m.cumsum(axis=1)
        final = cars[:, -1]
        out[k] = {"事件数": len(arr), "CAR[-5,+20]": final.mean(),
                  "t": final.mean() / final.std() * np.sqrt(len(final)),
                  "负CAR占比": (final < 0).mean(),
                  "CAR[0,+5]": (cars[:, 10] - cars[:, 4]).mean()}
    tab2 = pd.DataFrame(out).T.round(4)
    print("\n== (5) 处罚公告事件研究 (市场调整CAR) ==")
    print(tab2.to_string())
    tab2.to_csv(config.OUT_DIR / "penalty_event_study.csv",
                encoding="utf-8-sig")
