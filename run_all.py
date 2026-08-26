# -*- coding: utf-8 -*-
"""一键复现全部结果: python run_all.py  (逐个运行 analysis/01..06)"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = sorted((ROOT / "analysis").glob("0*.py"))

if __name__ == "__main__":
    for s in SCRIPTS:
        print(f"\n{'='*70}\n>>> {s.name}\n{'='*70}", flush=True)
        r = subprocess.run([sys.executable, str(s)], cwd=ROOT)
        if r.returncode != 0:
            print(f"!! {s.name} 运行失败 (exit {r.returncode}), 继续后续脚本",
                  flush=True)
    print("\n全部脚本运行完成, 结果见 output/")
