# -*- coding: utf-8 -*-
"""机制分析: 债务资本成本(cod)中介 (对应 机制分析/中介效应.do, 中介效应2.do).

修正原 do 两处不合理:
  [修正1] Bootstrap 由逐行独立重抽样改为按公司整群重抽样(cluster bootstrap),
          与面板数据的公司内相关结构一致.
  [修正2] 用预构建设计矩阵 + 最小二乘直接解 bootstrap 点估计(只需系数),
          1000 次重抽样秒级完成.
结论(如实): cod 的 a 路径不显著, Sobel 与 Bootstrap 均不支持该中介机制.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats
from greenwash import config
from greenwash.fe import reghdfe, esttab

CTRL = ["ESG_per", "Lev", "Dual", "TOP1", "TobinQ", "Dturn"]
X, M, Y = "GWS_industry", "cod", "ROA"


def design(d, cols):
    """常数 + 变量 + 行业/年份哑变量 的设计矩阵."""
    mats = [np.ones((len(d), 1)), d[cols].to_numpy(float)]
    for a in ("industry", "year"):
        mats.append(pd.get_dummies(d[a], drop_first=True, dtype=float).to_numpy())
    return np.hstack(mats)


if __name__ == "__main__":
    df = pd.read_stata(config.PANEL_MED)

    step = lambda y, xv, lab: reghdfe(df, y, xv, absorb=["industry", "year"],
                                      cluster=["industry", "year"], label=lab)
    m1 = step(Y, [X] + CTRL, "(1) ROA")
    m2 = step(M, [X] + CTRL, "(2) cod")
    m3 = step(Y, [M, X] + CTRL, "(3) ROA|cod")
    tab = esttab([m1, m2, m3])
    print("== 中介效应三步法 ==\n", tab.to_string(), "\n")
    tab.to_csv(config.OUT_DIR / "mediation_three_step.csv", encoding="utf-8-sig")

    # Sobel 检验
    a, sa = m2["table"].loc[X, ["coef", "se"]]
    b, sb = m3["table"].loc[M, ["coef", "se"]]
    z = a * b / np.sqrt(b**2 * sa**2 + a**2 * sb**2)
    print(f"Sobel: 间接效应 a*b={a*b:.6f}, z={z:.3f}, "
          f"p={2*(1-stats.norm.cdf(abs(z))):.4f}")

    # 按公司整群 Bootstrap (reps=1000, seed=12345)
    d = df[[Y, M, X] + CTRL + ["industry", "year", "stkcd"]].dropna()
    d = d.reset_index(drop=True)
    X2, y2 = design(d, [X] + CTRL), d[M].to_numpy(float)      # cod ~ GWS
    X3, y3 = design(d, [M, X] + CTRL), d[Y].to_numpy(float)   # ROA ~ cod + GWS
    firm_rows = {k: v.to_numpy() for k, v in d.groupby("stkcd").groups.items()}
    firms = np.array(list(firm_rows))
    rng = np.random.default_rng(12345)
    boots = []
    for _ in range(1000):
        idx = np.concatenate([firm_rows[f] for f in
                              rng.choice(firms, len(firms), replace=True)])
        ca = np.linalg.lstsq(X2[idx], y2[idx], rcond=None)[0][1]  # a: GWS 系数
        cb = np.linalg.lstsq(X3[idx], y3[idx], rcond=None)[0][1]  # b: cod 系数
        boots.append(ca * cb)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"Cluster Bootstrap(1000): 间接效应均值={boots.mean():.6f}, "
          f"95%CI=[{lo:.6f}, {hi:.6f}], 区间{'不含' if lo > 0 or hi < 0 else '含'}0")
