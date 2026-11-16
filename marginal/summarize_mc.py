#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap

bootstrap.add()

import argparse
from marginal_mc.reporting import summarize

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rebuild summaries from raw Monte Carlo checkpoints")
    p.add_argument("output")
    args = p.parse_args()
    summarize(args.output)
