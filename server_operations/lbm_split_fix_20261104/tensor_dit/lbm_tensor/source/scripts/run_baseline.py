#!/usr/bin/env python3
"""Run one baseline action; no implicit job submission."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flow_baselines.cli import main

if __name__=='__main__':
    main()
