"""Put the archived packages on sys.path.

Shared library: code_final/marginal_mc
Marginal drivers: code_final/marginal
Conditional package and drivers: code_final/conditional
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def add() -> None:
    for path in (ROOT, ROOT / "marginal", ROOT / "conditional"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
