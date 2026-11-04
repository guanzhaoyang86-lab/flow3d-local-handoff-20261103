#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from flow_observation.dps.audit import verify_prior, verify_posterior

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit actual completed DPS artifacts after the launcher returns')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prior-dir', type=Path)
    mode.add_argument('--posterior', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-particles', type=int, default=2)
    parser.add_argument('--num-samples', type=int, default=2)
    args = parser.parse_args()
    result = verify_prior(args.prior_dir, args.epochs) if args.prior_dir else verify_posterior(args.posterior, args.num_particles, args.num_samples)
    print(json.dumps(result))
