#!/usr/bin/env python3
"""Sample velocity fields using a flow prior and particle trajectory likelihood."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from flow_observation.dps.inference import parser, sample

if __name__ == '__main__':
    sample(parser().parse_args())
