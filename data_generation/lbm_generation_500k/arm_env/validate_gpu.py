"""Validate real CUDA execution and one original 32^3 LBM case before production."""
from __future__ import annotations
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import taichi as ti
import torch

from check_build import check_build, file_sha256, reusable_validation_marker


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
sys.path.insert(0, str(PACKAGE))
import run_generation as generation


def main():
    if platform.machine() != "aarch64":
        raise RuntimeError("This validation is for the DeltaAI ARM environment")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU is required")
    build = check_build(ROOT)
    # Explicit CUDA with fallback disabled: a CPU result cannot pass this gate.
    ti.init(arch=ti.cuda, enable_fallback=False, default_fp=ti.f32, offline_cache=False)
    values = ti.field(dtype=ti.f32, shape=256)

    @ti.kernel
    def fill():
        for index in values:
            values[index] = index * 0.25 + 1.0

    fill()
    ti.sync()
    if ti.lang.impl.current_cfg().arch != ti.cuda:
        raise RuntimeError("Taichi did not select its CUDA backend")
    np.testing.assert_array_equal(values.to_numpy(), np.arange(256, dtype=np.float32)*.25+1)

    # Exercise the repaired intrinsic itself, including data that differ only
    # in their high 32 bits. A stub or truncated operand cannot pass this test.
    matched_lanes = ti.field(dtype=ti.u32, shape=(2, 32))

    @ti.kernel
    def match_any_i64():
        ti.loop_config(block_dim=32)
        for lane in range(32):
            value = ti.cast(lane % 4, ti.i64)
            matched_lanes[0, lane] = ti.lang.impl.call_internal(
                "cuda_match_any_sync_i64", ti.u32(0xffffffff), value,
                with_runtime_context=False)
            matched_lanes[1, lane] = ti.lang.impl.call_internal(
                "cuda_match_any_sync_i64", ti.u32(0xffffffff),
                (value << 32) | ti.i64(7), with_runtime_context=False)

    match_any_i64()
    ti.sync()
    expected_masks = np.asarray([0x11111111 << (lane % 4) for lane in range(32)], dtype=np.uint32)
    np.testing.assert_array_equal(matched_lanes.to_numpy(), np.stack([expected_masks, expected_masks]))
    ti.reset()

    source = PACKAGE/"source"
    upstream = PACKAGE/"upstream/taichi_LBM3D"
    generation.check_upstream(upstream)
    collection = generation.load_collection(source)
    pilot_dir = ROOT/"pilot"
    pilot_dir.mkdir(exist_ok=True)
    args = collection._build_parser().parse_args(generation.collection_arguments(source, upstream, pilot_dir))
    case = collection.plan_cases(args)[0]
    archive = pilot_dir/f"{case.case_id}.npz"
    command = collection.build_generator_command(args, case, archive)
    if not archive.exists():
        with (pilot_dir/"generation.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    record = generation.validate_archive(archive, collection, args, case)
    with np.load(archive, allow_pickle=False) as data:
        assert data["flow_field"].shape == (1,3,32,32,32)
        assert data["trajectories_2d"].shape == (1,1,160,20,2)
        assert data["observation_mask"].all()
        speed = np.linalg.norm(data["flow_field"][0], axis=0)
        if not np.isfinite(speed).all() or float(speed.max()) <= 0:
            raise RuntimeError("Degenerate pilot velocity field")
    result = dict(
        schema_version=1, profile="deltaai-arm-llvm20",
        system=platform.system(), machine=platform.machine(),
        python_executable=sys.executable, python_version=platform.python_version(),
        taichi_version=version("taichi"), taichi_commit=ti._lib.core.get_commit_hash(),
        taichi_source_commit=build["taichi_source_commit"],
        torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0), llvm_version=ti._lib.core.get_llvm_target_support(),
        compiler_patch_sha256=build["compiler_patch_sha256"],
        cuda_kernel_validated=True, cuda_match_any_i64_validated=True, lbm32_pilot_validated=True,
        pilot_archive=str(archive), pilot_sha256=file_sha256(archive),
        pilot_record=record, pilot_max_speed=float(speed.max()),
        build_report_sha256=file_sha256(ROOT/"build_report.json"),
    )
    previous = reusable_validation_marker(ROOT, result, generation.runtime_provenance)
    if previous is not None:
        print("Reusing the existing validated environment marker after CUDA and pilot checks")
        print(json.dumps(previous, indent=2))
        return
    result.update(
        validated_at_utc=datetime.now(timezone.utc).isoformat(),
        validation_slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    marker=ROOT/"validated_env.json"
    temporary=marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(result,indent=2)+"\n")
    temporary.replace(marker)
    # Recheck the marker against the actual runtime using the production gate.
    generation.runtime_provenance("deltaai-arm-llvm20", marker)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
