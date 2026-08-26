# -*- coding: utf-8 -*-
"""多评级源 ESG 快照存档 (akshare, macro 环境运行).

来源: 华证(含E/S/G分项) / 秩鼎 / MSCI —— 均为**当前快照**, 非点时历史.
存档目的: ① 当期截面做评级分歧度; ② 逐期存档积累点时数据资产.
用法: python crawler/fetch_esg_snapshots.py
"""
import datetime as dt
from pathlib import Path

import akshare as ak

OUT = Path(__file__).resolve().parents[1] / "data_ext" / "esg_snapshots"
OUT.mkdir(parents=True, exist_ok=True)
TAG = dt.date.today().strftime("%Y%m%d")

FETCHERS = {"hz": ak.stock_esg_hz_sina,       # 华证
            "zd": ak.stock_esg_zd_sina,       # 秩鼎
            "msci": ak.stock_esg_msci_sina}   # MSCI

if __name__ == "__main__":
    for name, fn in FETCHERS.items():
        f = OUT / f"{name}_{TAG}.parquet"
        if f.exists():
            print(f"{name}: 已存在 {f.name}, 跳过")
            continue
        df = fn()
        df.to_parquet(f)
        print(f"{name}: {df.shape} -> {f.name}")
