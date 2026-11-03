#!/bin/bash
# Submit ARM build, CUDA validation/chunk 0, remaining chunks, and finalization.
set -euo pipefail

lbm500k_resume_failed=0
case "${1:-}" in
    '') ;;
    --resume-failed) lbm500k_resume_failed=1; shift ;;
    *) printf 'Usage: bash submit_arm.sh [--resume-failed]\n' >&2; exit 2 ;;
esac
[[ $# == 0 ]] || { printf 'Unexpected arguments.\n' >&2; exit 2; }
export LBM500K_ARM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
lbm500k_root="$(cd -- "$LBM500K_ARM_DIR/.." && pwd -P)"
[[ "$(uname -m)" == aarch64 ]] || {
    printf 'Run this on DeltaAI (gh-login): the prepared Taichi LLVM20 environment requires aarch64.\n' >&2
    exit 1
}
command -v sbatch >/dev/null
command -v flock >/dev/null
export LBM500K_PACKAGE_ROOT="$lbm500k_root"
export LBM500K_OUTPUT_DIR="${LBM500K_OUTPUT_DIR:-/work/hdd/biup/zguan2/datasets/lbm32_xz_500k_deltaai_arm_20261023}"
export LBM500K_BASE_PYTHON="${LBM500K_BASE_PYTHON:-/sw/user/python/miniforge3-pytorch-2.10.0/bin/python}"
LBM500K_PYTHON="$LBM500K_BASE_PYTHON"
[[ -x "$LBM500K_PYTHON" ]] || { printf 'Missing generation Python: %s\n' "$LBM500K_PYTHON" >&2; exit 1; }
[[ -d "$lbm500k_root/source/src/flow_observation" ]] || { printf 'Missing frozen source snapshot.\n' >&2; exit 1; }
mkdir -p "$LBM500K_ARM_DIR/submission" "$LBM500K_ARM_DIR/logs"
exec 9>"$LBM500K_ARM_DIR/submission/.submit.lock"
flock -n 9 || { printf 'Another submission process is running.\n' >&2; exit 1; }
if [[ -f "$LBM500K_ARM_DIR/submission/in-progress.json" || -f "$LBM500K_ARM_DIR/submission/partial.json" ]]; then
    printf 'An interrupted or partial submission exists. Inspect submission/in-progress.json and partial.json, and verify the recorded jobs with squeue/sacct before any manual recovery. Automatic resubmission is refused.\n' >&2
    exit 1
fi
if [[ -f "$LBM500K_ARM_DIR/submission/submission.json" && "$lbm500k_resume_failed" == 0 ]]; then
    printf 'This pipeline was already submitted. Recorded jobs:\n'
    "$LBM500K_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d["jobs"],indent=2))' "$LBM500K_ARM_DIR/submission/submission.json"
    exit 0
fi
if [[ "$lbm500k_resume_failed" == 1 ]]; then
    [[ -f "$LBM500K_ARM_DIR/submission/submission.json" ]] || {
        printf '%s\n' '--resume-failed requires a complete recorded pipeline.' >&2
        exit 1
    }
    # Absence from squeue alone is insufficient: require accounting terminal states too.
    "$LBM500K_PYTHON" - "$LBM500K_ARM_DIR/submission/submission.json" "$LBM500K_OUTPUT_DIR" "$LBM500K_ARM_DIR" <<'PY'
import getpass,json,pathlib,re,subprocess,sys
try:
    record=json.loads(pathlib.Path(sys.argv[1]).read_text())
    jobs=record["jobs"]
    ids=[str(jobs[key]) for key in ("build_environment","validate_first_chunk","remaining_chunks","finalize")]
    if len(set(ids)) != 4 or not all(re.fullmatch(r"[0-9]+", value) for value in ids):
        raise ValueError("Recorded pipeline is incomplete or has invalid job IDs")
    output=pathlib.Path(sys.argv[2]).expanduser().resolve()
    if output != pathlib.Path(record["output_dir"]).expanduser().resolve():
        raise ValueError("Recovery must use the original output directory")
    if pathlib.Path(sys.argv[3]).expanduser().resolve() != pathlib.Path(record["arm_dir"]).expanduser().resolve():
        raise ValueError("Recovery must use the original ARM environment directory")
    if (output/"COMPLETE.json").exists():
        raise ValueError("Dataset is already complete; use its manifest")
    active=subprocess.check_output(["squeue","--noheader","--user",getpass.getuser(),"--format=%A"],text=True)
    if set(active.split()) & set(ids):
        raise ValueError("A recorded job is still active or pending in squeue; recovery refused")
    accounting=subprocess.check_output(["sacct","--noheader","--parsable2","--allocations",
        "--jobs="+",".join(ids),"--format=JobID%80,State%40"],text=True)
    terminal={"BOOT_FAIL","CANCELLED","COMPLETED","DEADLINE","FAILED","NODE_FAIL",
              "OUT_OF_MEMORY","PREEMPTED","REVOKED","TIMEOUT"}
    seen=set(); failed=False
    for line in accounting.splitlines():
        columns=line.strip().split("|")
        if len(columns)<2:
            continue
        root=columns[0].strip().split("_",1)[0].split(".",1)[0]
        if root not in ids:
            continue
        state=columns[1].strip().split()[0].rstrip("+")
        if state not in terminal:
            raise ValueError("A recorded job has nonterminal accounting state: "+line)
        seen.add(root); failed |= state != "COMPLETED"
    if seen != set(ids):
        raise ValueError("Accounting does not yet confirm all four recorded jobs terminated")
    if not failed:
        raise ValueError("Recorded pipeline has no failed jobs; recovery refused")
except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
    raise SystemExit("Cannot safely resume: "+str(error))
PY
fi

# No solver, Torch import or GPU computation is performed by this plan check.
"$LBM500K_PYTHON" -B "$lbm500k_root/run_generation.py" \
    --source-root "$lbm500k_root/source" \
    --upstream-repo "$lbm500k_root/upstream/taichi_LBM3D" \
    --output-dir "$LBM500K_OUTPUT_DIR" --num-chunks 25 \
    --runtime-profile deltaai-arm-llvm20 --environment-manifest "$LBM500K_ARM_DIR/validated_env.json" --check-plan \
    > "$LBM500K_ARM_DIR/submission/checked_plan.json"

lbm500k_common=(--account=biup-dtai-gh
    --nodes=1 --ntasks=1 --cpus-per-task=8
    --no-requeue --kill-on-invalid-dep=yes --chdir="$LBM500K_ARM_DIR" --export=ALL
    --output="$LBM500K_ARM_DIR/logs/%x_%j.out" --error="$LBM500K_ARM_DIR/logs/%x_%j.err")
lbm500k_build_args=(--partition=ghx4-interactive --mem=48G --gpus-per-node=1 --time=02:00:00)
lbm500k_gpu=(--partition=ghx4 --mem=32G --gpus-per-node=1)
for lbm500k_script in build.slurm validate_gpu.slurm generate_arm.slurm finalize_arm.slurm; do
    [[ -f "$LBM500K_ARM_DIR/$lbm500k_script" ]] || { printf 'Missing ARM job script: %s\n' "$lbm500k_script" >&2; exit 1; }
done
# The site requires a GPU allocation even for the compiler phase.
# One interactive job builds and validates; ordinary ghx4 jobs generate the dataset.
sbatch --test-only "${lbm500k_common[@]}" "${lbm500k_build_args[@]}" "$LBM500K_ARM_DIR/build.slurm"
sbatch --test-only "${lbm500k_common[@]}" "${lbm500k_gpu[@]}" --time=02:00:00 "$LBM500K_ARM_DIR/validate_gpu.slurm"

lbm500k_build=''
lbm500k_job0=''
lbm500k_array=''
lbm500k_final=''
lbm500k_phase='before-build-submission'
lbm500k_previous_record=''
lbm500k_record() {
    "$LBM500K_PYTHON" - "$LBM500K_ARM_DIR/submission" "$LBM500K_OUTPUT_DIR" "$LBM500K_ARM_DIR" "$lbm500k_phase" "$lbm500k_previous_record" "$lbm500k_build" "$lbm500k_job0" "$lbm500k_array" "$lbm500k_final" <<'PY'
import datetime,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
data={"submitted_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "cluster":"DeltaAI", "account":"biup-dtai-gh",
      "partitions_per_stage":{"build_environment":"ghx4-interactive","validate_first_chunk":"ghx4","remaining_chunks":"ghx4","finalize":"ghx4"},
      "gpus_per_stage":{"build_environment":1,"validate_first_chunk":1,"remaining_chunks":1,"finalize":1},"array_concurrency":1,"output_dir":sys.argv[2],
       "arm_dir":sys.argv[3],"runtime_profile":"deltaai-arm-llvm20","phase":sys.argv[4],"previous_record":sys.argv[5] or None,
      "physical_trajectories":500000,"projection":"xz","grid_shape":[32,32,32],
       "jobs":dict(zip(("build_environment","validate_first_chunk","remaining_chunks","finalize"),sys.argv[6:]))}
for name in ("in-progress.json","partial.json"):
    path=root/name
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)
PY
}
# Publish a guard before the first sbatch call can be accepted without a saved ID.
lbm500k_record
trap 'lbm500k_status=$?; if [[ "$lbm500k_status" != 0 ]]; then printf "Submission interrupted. Preserve submission/in-progress.json and partial.json; verify squeue/sacct before manual recovery.\n" >&2; fi' EXIT
if [[ "$lbm500k_resume_failed" == 1 ]]; then
    lbm500k_previous_record="$("$LBM500K_PYTHON" - "$LBM500K_ARM_DIR/submission" <<'PY'
import datetime,pathlib,shutil,sys,uuid
root=pathlib.Path(sys.argv[1]); history=root/"attempts"; history.mkdir(exist_ok=True)
path=history/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.json')
shutil.copyfile(root/'submission.json',path)
print(path)
PY
)"
    lbm500k_record
fi
lbm500k_build="$(sbatch --parsable "${lbm500k_common[@]}" "${lbm500k_build_args[@]}" --job-name=lbm500k-arm-build "$LBM500K_ARM_DIR/build.slurm")"
lbm500k_build="${lbm500k_build%%;*}"
[[ "$lbm500k_build" =~ ^[0-9]+$ ]] || { printf 'Invalid build job ID.\n' >&2; exit 1; }
lbm500k_phase='before-validation-submission'
lbm500k_record
lbm500k_job0="$(sbatch --parsable "${lbm500k_common[@]}" "${lbm500k_gpu[@]}" --time=02:00:00 --job-name=lbm500k-arm-validate \
    --dependency="afterok:$lbm500k_build" "$LBM500K_ARM_DIR/validate_gpu.slurm")"
lbm500k_job0="${lbm500k_job0%%;*}"
[[ "$lbm500k_job0" =~ ^[0-9]+$ ]] || { printf 'Invalid validation job ID; see submission/partial.json.\n' >&2; exit 1; }
lbm500k_phase='before-array-submission'
lbm500k_record
lbm500k_array="$(sbatch --parsable "${lbm500k_common[@]}" "${lbm500k_gpu[@]}" --time=02:00:00 --job-name=lbm500k-arm-part \
    --array=1-24%1 --dependency="afterok:$lbm500k_job0" "$LBM500K_ARM_DIR/generate_arm.slurm")"
lbm500k_array="${lbm500k_array%%;*}"
[[ "$lbm500k_array" =~ ^[0-9]+$ ]] || { printf 'Invalid array job ID; see submission/partial.json.\n' >&2; exit 1; }
lbm500k_phase='before-finalize-submission'
lbm500k_record
lbm500k_final="$(sbatch --parsable "${lbm500k_common[@]}" "${lbm500k_gpu[@]}" --time=00:30:00 --job-name=lbm500k-arm-check \
    --dependency="afterok:$lbm500k_array" "$LBM500K_ARM_DIR/finalize_arm.slurm")"
lbm500k_final="${lbm500k_final%%;*}"
[[ "$lbm500k_final" =~ ^[0-9]+$ ]] || { printf 'Invalid finalize job ID; see submission/partial.json.\n' >&2; exit 1; }
lbm500k_phase='submitted'
lbm500k_record
mv -- "$LBM500K_ARM_DIR/submission/partial.json" "$LBM500K_ARM_DIR/submission/submission.json"
rm -- "$LBM500K_ARM_DIR/submission/in-progress.json"
printf 'Submitted build=%s validation=%s array=%s finalize=%s\n' "$lbm500k_build" "$lbm500k_job0" "$lbm500k_array" "$lbm500k_final"
printf 'Dataset: %s\n' "$LBM500K_OUTPUT_DIR"
