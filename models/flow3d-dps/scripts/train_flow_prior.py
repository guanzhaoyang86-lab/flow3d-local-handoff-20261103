#!/usr/bin/env python3
"""Train a flow-only, N-independent diffusion prior."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from flow_observation.dps.training import parser, train

if __name__ == '__main__':
    train(parser().parse_args())
