# -*- coding: utf-8 -*-
"""巨潮公告 "处罚" 关键词逐股爬取 (akshare→cninfo, macro 环境运行).

对 ESG 样本全部股票检索 2014-2024 标题含"处罚"的公司公告,
得到 处罚公告日期+标题 → 用于硬行为漂绿旗标(13)与曝光事件研究.
断点续爬: 已完成代码记录在 done.txt, 重跑自动跳过.
用法: python crawler/fetch_cninfo_penalty.py
"""
import re
import time
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_ext"
OUT.mkdir(exist_ok=True)
RESULT = OUT / "cninfo_penalty_2014_2024.parquet"
DONE = OUT / "cninfo_penalty_done.txt"


def universe():
    df = pd.read_stata(ROOT.parent / "大创" / "完全11.dta")
    codes = sorted({f"{int(c):06d}" for c in df["stkcd"].dropna().unique()})
    return [c for c in codes if c[0] in "036"]


if __name__ == "__main__":
    codes = universe()
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    rows = [pd.read_parquet(RESULT)] if RESULT.exists() else []
    todo = [c for c in codes if c not in done]
    print(f"股票总数 {len(codes)}, 待爬 {len(todo)}")

    for i, c in enumerate(todo):
        for attempt in (1, 2):
            try:
                df = ak.stock_zh_a_disclosure_report_cninfo(
                    symbol=c, market="沪深京", keyword="处罚",
                    start_date="20140101", end_date="20241231")
                if len(df):
                    df["公告标题"] = df["公告标题"].str.replace(
                        r"</?em>", "", regex=True)
                    df["symbol"] = c
                    rows.append(df)
                done.add(c)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"{c} FAIL: {repr(e)[:80]}")
                time.sleep(3)
        time.sleep(0.25)
        if (i + 1) % 100 == 0 or i == len(todo) - 1:
            if rows:
                pd.concat(rows, ignore_index=True).drop_duplicates(
                    subset=["symbol", "公告标题", "公告时间"]).to_parquet(RESULT)
            DONE.write_text("\n".join(sorted(done)))
            print(f"进度 {i+1}/{len(todo)}, 累计公告 "
                  f"{sum(len(r) for r in rows)}", flush=True)
    print("完成")
