# -*- coding: utf-8 -*-
"""让 analysis/ 脚本无需安装即可 import src/greenwash."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
