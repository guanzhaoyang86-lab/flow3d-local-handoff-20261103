#!/usr/bin/env python3
"""Refresh the ARM pipeline's progress from Slurm and validated JSON records.

CPU only. No solver, Torch, Taichi, submission, cancellation, or NPZ counting.
Prints JSON by default; --write atomically replaces the progress file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
STAGES = ("build_environment", "validate_first_chunk", "remaining_chunks", "finalize")
FAILURES = {"BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT"}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scheduler_snapshot(jobs: dict, runner=subprocess.run, *,
                       execution_mode: str = "staged") -> tuple[dict, list[str]]:
    """Prefer live queue states; accounting supplies completed/failed jobs."""
    ids = [str(jobs[key]) for key in STAGES]
    if not all(re.fullmatch(r"[0-9]+", item) for item in ids):
        raise ValueError("Submission must contain numeric job IDs")
    if execution_mode == "continuous":
        if ids[2] != ids[3] or len(set(ids)) != 3:
            raise ValueError("Continuous submission requires distinct build, validation and production IDs; "
                             "only remaining_chunks and finalize may share an ID")
    elif execution_mode == "staged":
        if len(set(ids)) != len(STAGES):
            raise ValueError("Submission must contain four distinct numeric job IDs")
    else:
        raise ValueError(f"Unknown execution_mode: {execution_mode!r}")
    ids = list(dict.fromkeys(ids))
    snapshots, errors = {}, []
    commands = (
        ("accounting", ["sacct", "--noheader", "--parsable2", "--allocations",
                        "--jobs=" + ",".join(ids), "--format=JobID%100,State%40,ExitCode"]),
        # Completed jobs may no longer be known to squeue. Query this user's
        # live jobs and filter below, rather than passing expired job IDs and
        # turning their normal absence into an "Invalid job id" error.
        ("queue", ["squeue", "--noheader", "--array", "--user=" + getpass.getuser(),
                   "--format=%i|%T|%R|%S"]),
    )
    for source, command in commands:
        try:
            result = runner(command, capture_output=True, text=True, check=True, timeout=30)
            for line in result.stdout.splitlines():
                fields = [item.strip() for item in line.split("|")]
                if len(fields) < (4 if source == "queue" else 3):
                    continue
                match = re.fullmatch(r"([0-9]+)(?:_(?:[0-9]+|\[[0-9,:%-]+\]))?", fields[0])
                if not match or match.group(1) not in ids:
                    continue  # Ignore .batch/.extern steps and unrelated jobs.
                entry = {"state": fields[1].split()[0].rstrip("+"), "source": source}
                if source == "queue":
                    entry.update(reason_or_node=fields[2], start_prediction=fields[3])
                else:
                    entry["exit_code"] = fields[2]
                snapshots[fields[0]] = entry
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"{source}: {error}")
    return snapshots, errors


def validated_chunk_counts(output: Path, total=3125, num_chunks=25) -> tuple[int, list[dict], list[str]]:
    """Count only per-case progress persisted AFTER validate_archive succeeds."""
    cases, records, errors = 0, [], []
    for path in sorted((output / "chunks").glob("chunk_*.json")):
        try:
            record = read_json(path)
            index = record["chunk_index"]
            if type(index) is not int or not 0 <= index < num_chunks:
                raise ValueError("Invalid chunk index")
            start, stop = total * index // num_chunks, total * (index + 1) // num_chunks
            if path.name != f"chunk_{index:03d}.json":
                raise ValueError("Filename and chunk index differ")
            if (record["case_start"], record["case_stop"], record["num_cases"]) != (start, stop, stop-start):
                raise ValueError("Chunk range differs from the fixed, non-overlapping plan")
            count = record.get("completed_cases", 0)
            generated, reused = record.get("generated", 0), record.get("reused", 0)
            if any(type(value) is not int or value < 0 for value in (count, generated, reused)):
                raise ValueError("Invalid per-case counts")
            if count > stop-start or generated+reused != count:
                raise ValueError("Inconsistent per-case counts")
            if record["status"] not in {"running", "completed", "failed"}:
                raise ValueError("Unknown chunk status")
            if record["status"] == "completed" and count != stop-start:
                raise ValueError("Completed chunk has an incomplete count")
            cases += count
            records.append({"chunk_index": index, "status": record["status"], "completed_cases": count,
                            "num_cases": stop-start, "slurm_job_id": record.get("slurm_job_id")})
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{path.name}: {error}")
    return cases, records, errors


def environment_markers(arm: Path) -> tuple[dict, list[str]]:
    status = {"taichi_build_completed": False, "cuda_and_lbm32_validated": False}
    errors = []
    try:
        report_path = arm / "build_report.json"
        if not report_path.exists():
            return status, errors
        report = read_json(report_path)
        patch_sha = digest(arm / "compiler.patch")
        if report.get("status") != "compiled_and_installed" or report.get("compiler_patch_sha256") != patch_sha:
            raise ValueError("Build report does not certify the current compiler patch")
        status["taichi_build_completed"] = True
        marker_path = arm / "validated_env.json"
        if marker_path.exists():
            marker = read_json(marker_path)
            if not (marker.get("cuda_kernel_validated") is True
                    and marker.get("lbm32_pilot_validated") is True
                    and marker.get("compiler_patch_sha256") == patch_sha
                    and marker.get("build_report_sha256") == digest(report_path)):
                raise ValueError("GPU marker does not certify the current build report")
            status["cuda_and_lbm32_validated"] = True
    except (OSError, ValueError) as error:
        errors.append(str(error))
    return status, errors


def completion_marker(output: Path, total=3125, tracks=500000) -> tuple[bool, list[str]]:
    path = output / "COMPLETE.json"
    if not path.exists():
        return False, []
    try:
        marker = read_json(path)
        if not (marker.get("status") == "completed" and marker.get("num_cases") == total
                and marker.get("physical_trajectories") == tracks
                and marker.get("single_view_trajectories") == tracks
                and marker.get("num_independent_flow_groups") == total
                and marker.get("projection_names") == ["xz"]):
            raise ValueError("Final count or projection differs from the requested dataset")
        if marker.get("dataset_version") != digest(output / "SHA256SUMS"):
            raise ValueError("Final checksum-list digest differs from COMPLETE.json")
        if not (output / "manifest.json").is_file():
            raise ValueError("Final manifest is missing")
        return True, []
    except (OSError, ValueError) as error:
        return False, [f"COMPLETE.json: {error}"]


def refresh(previous: dict, submission: dict, snapshots: dict, scheduler_errors: list[str]) -> dict:
    output, arm = Path(submission["output_dir"]), Path(submission["arm_dir"])
    total, tracks_per_case = 3125, 160
    count, chunks, chunk_errors = validated_chunk_counts(output, total)
    env, env_errors = environment_markers(arm)
    complete, complete_errors = completion_marker(output, total, total*tracks_per_case)
    errors = scheduler_errors + chunk_errors + env_errors + complete_errors
    final = snapshots.get(str(submission["jobs"]["finalize"]), {})
    final_ok = final.get("state") == "COMPLETED" and final.get("exit_code") == "0:0"
    states = {record["state"] for record in snapshots.values()}
    if complete and final_ok and not errors:
        status, count = "completed", total
    elif errors:
        status = "monitoring_incomplete"
    elif states & FAILURES:
        status = "failed"
    elif "RUNNING" in states or "COMPLETING" in states:
        status = "running_generation" if env["cuda_and_lbm32_validated"] else "running_arm_build_or_validation"
    elif "PENDING" in states:
        status = "waiting_for_generation_allocation" if env["cuda_and_lbm32_validated"] else "waiting_for_arm_build_allocation"
    else:
        status = "awaiting_scheduler_accounting"
    result = dict(previous)
    result.update(status=status, updated_at_utc=datetime.now(timezone.utc).isoformat(), submission=submission,
                  last_observed_queue=snapshots, completed_production_cases=count,
                  completed_physical_tracks=count*tracks_per_case, dataset_dir=str(output),
                  chunk_progress=chunks, progress_errors=errors)
    result["environment"] = {**previous.get("environment", {}), **env}
    result["completion_condition"] = "Validated COMPLETE.json counts/checksum list and final Slurm allocation COMPLETED with exit code 0:0."
    result["notes"] = ["Production counts come from per-case validated chunk records, never NPZ file counts.",
                       "Start predictions are scheduler estimates, in the cluster's local time.",
                       "Only successful compiler and GPU validation release dependent production stages."]
    return result


def atomic_write(path: Path, value: dict) -> None:
    handle, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=ROOT/"arm_env/submission/submission.json")
    parser.add_argument("--progress", type=Path, default=ROOT/"progress.json")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    submission = read_json(args.submission)
    previous = read_json(args.progress) if args.progress.exists() else {}
    snapshot, errors = scheduler_snapshot(
        submission["jobs"], execution_mode=submission.get("execution_mode", "staged"))
    value = refresh(previous, submission, snapshot, errors)
    if args.write:
        atomic_write(args.progress, value)
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
