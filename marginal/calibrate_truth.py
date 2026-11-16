#!/usr/bin/env python3
"""Standalone entry point for the supplied oracle calibration module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap

bootstrap.add()

from marginal_mc.oracle_calibration import main
if __name__ == "__main__":
    main()
