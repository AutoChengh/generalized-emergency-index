#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility wrapper for the GEI GIF visualization command."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gei.visualization import main


if __name__ == "__main__":
    main()
