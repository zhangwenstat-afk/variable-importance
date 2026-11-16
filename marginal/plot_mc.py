#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap

bootstrap.add()

import argparse
from marginal_mc.plotting import make_plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Monte Carlo summaries as separate PNG figures")
    parser.add_argument("output")
    args = parser.parse_args()
    make_plots(args.output)
