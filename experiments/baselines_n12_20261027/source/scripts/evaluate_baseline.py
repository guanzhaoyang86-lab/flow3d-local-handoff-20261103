#!/usr/bin/env python
"""Evaluate an explicit set of held-out cases, with verified case-wise resume.

This script creates local evaluation programs only; it does not submit jobs.
Each prediction is fixed before its reference is used for posthoc evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from flow_baselines.config import MODELS, resolve_config
from flow_baselines.inference import predict
from flow_baselines.reporting import verify
from flow_observation.dps.data import sha256
from flow_observation.sparse_dataset import load_sparse_flow_manifest


def parse_case_indices(value: str, size: int) -> list[int]:
    """Preserve an explicit user's case order, and reject duplicates or gaps."""
    if size < 1:
        raise ValueError("The requested evaluation split contains no cases")
    if value.strip().lower() == "all":
        return list(range(size))
    pieces = value.split(",")
    if any(not item.strip() for item in pieces):
        raise ValueError("case-indices must be all or a comma-separated integer list")
    try:
        indices = [int(item.strip()) for item in pieces]
    except ValueError as error:
        raise ValueError("case-indices must be all or a comma-separated integer list") from error
    if any(index < 0 or index >= size for index in indices):
        raise ValueError(f"case-indices must be between 0 and {size - 1}")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate case-indices are not allowed")
    return indices


def _read_json(path: Path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def _write_progress(path: Path, payload: dict) -> None:
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _request_config(args):
    overrides = _read_json(Path(args.config)) if args.config else {}
    if not isinstance(overrides, dict):
        raise ValueError("Config JSON must be an object containing partial baseline settings")
    overrides = deepcopy(overrides)
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.num_particles is not None:
        overrides["num_particles"] = args.num_particles
    if args.viscosity is not None:
        if "pinn" in overrides and not isinstance(overrides["pinn"], dict):
            raise ValueError("pinn config must be an object")
        overrides.setdefault("pinn", {})["viscosity"] = args.viscosity
    manifest = Path(args.manifest).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    kind = args.model
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != "flow3d-baseline-v1" or saved.get("manifest_sha256") != sha256(manifest):
            raise ValueError("Checkpoint format or manifest mismatch")
        if kind is not None and kind != saved.get("model"):
            raise ValueError("Checkpoint model differs from requested model")
        kind = saved["model"]
        config = resolve_config(overrides=saved["config"])
        # Keep precisely the same override restrictions as inference.predict.
        allowed = {"num_particles", "views", "boundary_projection", "num_samples", "num_probes",
                   "sampling_steps", "sampling_eta", "cfg_scale", "observation_seed", "sampling_seed",
                   "fit_steps", "fit_learning_rate", "sigma", "coefficient_weight", "depth_learning_rate", "seed"}
        for key, value in overrides.items():
            if value != config.get(key) and key not in allowed:
                raise ValueError(f"Cannot change fitted model configuration: {key}")
            config[key] = value
        config = resolve_config(overrides=config)
    else:
        config = resolve_config(overrides=overrides)
        if kind not in ("rbf", "pinn"):
            raise ValueError("Specify a fitted checkpoint, or use --model rbf/pinn")
    if kind not in MODELS:
        raise ValueError("Unknown baseline model")
    if kind == "pinn" and config["pinn"]["viscosity"] is None:
        raise ValueError("PINN requires known kinematic viscosity via --viscosity or config")
    if kind == "pinn" and config["boundary_projection"] == "known":
        raise ValueError("PINN does not support hard boundary postprojection")
    return manifest, checkpoint, kind, config, overrides


def _verify_existing(output: Path, *, case, request: dict) -> None:
    verify(output)
    record = _read_json(output.with_suffix(".json"))
    expected = {"manifest_sha256": request["manifest_sha256"],
                "checkpoint_sha256": request["checkpoint_sha256"],
                "model": request["model"], "config": request["config"],
                "split": request["split"], "case_id": case.case_id,
                "flow_group_id": case.flow_group_id, "source_case_sha256": sha256(case.path)}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Existing output differs from requested {key}: {output}")
    if not isinstance(record.get("evaluation"), dict):
        raise ValueError(f"Existing output has no posthoc evaluation: {output}")


def run_batch(args) -> dict:
    manifest, checkpoint, kind, config, overrides = _request_config(args)
    _, splits = load_sparse_flow_manifest(manifest)
    if args.split not in ("validation", "test"):
        raise ValueError("Batch evaluation split must be validation or test")
    indices = parse_case_indices(args.case_indices, len(splits[args.split]))
    output_dir = Path(args.output_dir).resolve()
    progress_path = output_dir / "progress.json"
    request = {"manifest": str(manifest), "manifest_sha256": sha256(manifest),
               "checkpoint": str(checkpoint) if checkpoint else None,
               "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
               "model": kind, "config": config, "split": args.split,
               "case_indices": indices, "device": args.device, "posthoc_evaluate": True}
    if progress_path.exists():
        if not args.resume:
            raise FileExistsError(f"Batch progress already exists; use --resume: {progress_path}")
        previous = _read_json(progress_path)
        if previous.get("format") != "flow3d-baseline-batch-v1" or previous.get("request") != request:
            raise ValueError("Existing batch request differs; choose a new output directory")
    rows = []
    # Validate all existing outputs before modifying progress or predicting any
    # case. A partial NPZ/JSON pair is a conflict, never an invitation to overwrite.
    for index in indices:
        case = splits[args.split][index]
        output = output_dir / f"case_{index:06d}.npz"
        exists = output.exists() or output.with_suffix(".json").exists()
        if exists:
            if not args.resume:
                raise FileExistsError(f"Existing case output; use --resume: {output}")
            if not output.is_file() or not output.with_suffix(".json").is_file():
                raise ValueError(f"Incomplete case output; move it aside before retrying: {output}")
            _verify_existing(output, case=case, request=request)
        rows.append({"case_index": index, "case_id": case.case_id, "output": str(output),
                     "status": "completed" if exists else "pending", "resumed": bool(exists)})
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = {"format": "flow3d-baseline-batch-v1", "status": "running", "request": request,
                "total_cases": len(rows), "completed_cases": sum(row["status"] == "completed" for row in rows),
                "failed_cases": 0, "cases": rows}
    _write_progress(progress_path, progress)
    for row in rows:
        if row["status"] == "completed":
            continue
        row["status"] = "running"
        _write_progress(progress_path, progress)
        try:
            index = row["case_index"]
            predict(manifest, row["output"], kind=kind, checkpoint=checkpoint, config=overrides,
                    split=args.split, case_index=index, device=args.device, evaluate=True)
            _verify_existing(Path(row["output"]), case=splits[args.split][index], request=request)
            row["status"] = "completed"
            progress["completed_cases"] += 1
        except Exception as error:
            row["status"] = "failed"
            row["error"] = {"type": type(error).__name__, "message": str(error)}
            progress["failed_cases"] += 1
        _write_progress(progress_path, progress)
    progress["status"] = "completed" if progress["completed_cases"] == len(rows) else "failed"
    _write_progress(progress_path, progress)
    return progress


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--checkpoint")
    result.add_argument("--model", choices=MODELS)
    result.add_argument("--config", help="Partial JSON settings; CLI seed/N/viscosity take precedence")
    result.add_argument("--split", choices=("validation", "test"), default="validation")
    result.add_argument("--case-indices", default="all", help="all or comma-separated zero-based split indices")
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--seed", type=int)
    result.add_argument("--num-particles", type=int)
    result.add_argument("--viscosity", type=float)
    result.add_argument("--resume", action="store_true", help="Reuse only verified, exactly matching NPZ/JSON pairs")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    progress = run_batch(args)
    print(json.dumps({key: progress[key] for key in ("status", "total_cases", "completed_cases", "failed_cases")}))
    return 0 if progress["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
