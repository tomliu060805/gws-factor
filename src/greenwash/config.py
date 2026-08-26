# -*- coding: utf-8 -*-
"""路径配置.

设计原则:
  1. 项目内部路径全部相对于仓库根目录(由本文件位置推导), 不含任何绝对路径;
  2. 项目外部数据(受版权保护的 ESG/财务面板、行情库)通过**环境变量**指定,
     并给出本机默认值, 便于在其他环境中复现.

可配置环境变量
--------------
GWS_DATA_DIR       ESG/财务年度面板所在目录(含 .dta 与无风险利率表)
                   默认: <repo>/../大创
GWS_MARKET_DIR     A股行情库根目录, 需包含
                     stock/price/price_daily/YYYY-MM-DD.parquet
                     stock/fundamental/valuation/YYYY-MM-DD.parquet
                     index/price/price_daily/YYYY-MM-DD.parquet
                   默认: <repo>/data/market
GWS_INDEX_WEIGHTS  指数成分权重目录, 含 index_weights_300/500.parquet
                   默认: <repo>/data/index_weights
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # 仓库根目录


def _env_path(var, default):
    return Path(os.environ.get(var, default)).expanduser()


# ---- 项目外部数据(不随仓库分发) ----
DATA_DIR = _env_path("GWS_DATA_DIR", ROOT.parent / "大创")
MARKET_DIR = _env_path("GWS_MARKET_DIR", ROOT / "data" / "market")
INDEX_WEIGHT_DIR = _env_path("GWS_INDEX_WEIGHTS", ROOT / "data" / "index_weights")

PANEL_FULL = DATA_DIR / "完全11.dta"                 # 9,835 firm-year, 2014-2023
PANEL_REG = DATA_DIR / "完全7_1行业年份、行业补充缺失值+异质性变量_已生成虚拟变量和滞后.dta"
PANEL_MED = DATA_DIR / "机制分析" / "机制分析 含COD.dta"
RF_FILE = DATA_DIR / "个股超额收益率R" / "月度无风险利率.xlsx"

PRICE_DIR = MARKET_DIR / "stock" / "price" / "price_daily"
VAL_DIR = MARKET_DIR / "stock" / "fundamental" / "valuation"
INDEX_DIR = MARKET_DIR / "index" / "price" / "price_daily"
W300 = INDEX_WEIGHT_DIR / "index_weights_300.parquet"
W500 = INDEX_WEIGHT_DIR / "index_weights_500.parquet"

# ---- 项目内部输出(相对路径) ----
OUT_DIR = ROOT / "output"
CHART_DIR = ROOT / "figures" / "v1_monthly_factor"
CHART_DIR_EN = ROOT / "figures" / "v3_paper_en"
PAPER_FIG_DIR = ROOT / "paper" / "figs"
DATA_EXT = ROOT / "data_ext"                        # 爬虫产出

for _d in (OUT_DIR, CHART_DIR, DATA_EXT):
    _d.mkdir(parents=True, exist_ok=True)
