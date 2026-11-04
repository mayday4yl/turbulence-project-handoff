#!/usr/bin/env python3
"""Run the previously verified receiver demo from the preserved archive."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / '03_归档资料/tools/receiver/demo.py'), run_name='__main__')
