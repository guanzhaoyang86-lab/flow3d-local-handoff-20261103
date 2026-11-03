"""Test the ARM submission wrapper with local scheduler stubs only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ARM = Path(__file__).resolve().parent
PACKAGE = ARM.parent


class ArmSubmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lbm500k_arm_submit_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.package = self.root / "package"
        self.arm = self.package / "arm_env"
        self.arm.mkdir(parents=True)
        shutil.copy2(ARM / "submit_arm.sh", self.arm / "submit_arm.sh")
        # Job bodies belong to the build/validation tests; scheduler stubs never execute them.
        for name in ("build.slurm", "validate_gpu.slurm", "generate_arm.slurm", "finalize_arm.slurm"):
            (self.arm / name).write_text("#!/bin/bash\nexit 99\n")
        for name in ("source", "upstream", "run_generation.py"):
            (self.package / name).symlink_to(PACKAGE / name, target_is_directory=name != "run_generation.py")
        binary = self.root / "bin"
        binary.mkdir()
        (binary / "uname").write_text('#!/bin/bash\nprintf "%s\\n" "${LBM500K_FAKE_MACHINE:-aarch64}"\n')
        sbatch = '''import json,os,pathlib,sys
path=pathlib.Path(os.environ['LBM500K_FAKE_LOG'])
args=sys.argv[1:]
calls=path.read_text().splitlines() if path.exists() else []
with path.open('a') as stream:stream.write(json.dumps(args)+'\\n')
if '--test-only' in args:raise SystemExit(0)
arm=pathlib.Path(os.environ['LBM500K_ARM_DIR'])
assert (arm/'submission/in-progress.json').is_file()
assert (arm/'submission/partial.json').is_file()
assert arm.is_absolute()
if pathlib.Path(args[-1]).name==os.environ.get('LBM500K_FAKE_FAIL_STAGE'):raise SystemExit(1)
if pathlib.Path(args[-1]).name==os.environ.get('LBM500K_FAKE_BAD_ID_STAGE'):
    print('unrecordable accepted response');raise SystemExit(0)
print(str(910000+len(calls))+';deltaai')
'''
        (binary / "sbatch").write_text(f"#!{sys.executable}\n" + sbatch)
        (binary / "squeue").write_text(f"#!{sys.executable}\nimport os\nprint(os.environ.get('LBM500K_FAKE_ACTIVE_JOBS',''))\n")
        sacct = '''import json,os,pathlib
jobs=json.loads(pathlib.Path(os.environ['LBM500K_FAKE_RECORD']).read_text())['jobs']
if os.environ.get('LBM500K_FAKE_ACCOUNTING_MISSING')=='1':raise SystemExit(0)
for key,value in jobs.items():
    state=os.environ.get('LBM500K_FAKE_ACCOUNTING_STATE','FAILED') if key=='build_environment' else 'CANCELLED'
    print(value+'|'+state+'|')
'''
        (binary / "sacct").write_text(f"#!{sys.executable}\n" + sacct)
        for path in binary.iterdir():
            path.chmod(0o755)
        self.log = self.root / "scheduler.jsonl"
        self.env = os.environ.copy()
        self.env.update(PATH=str(binary) + os.pathsep + self.env["PATH"],
                        LBM500K_FAKE_LOG=str(self.log),
                        LBM500K_FAKE_RECORD=str(self.arm / "submission/submission.json"),
                        LBM500K_BASE_PYTHON=sys.executable,
                        LBM500K_OUTPUT_DIR=str(self.root / "dataset"))

    def submit(self, *arguments):
        return subprocess.run(["bash", str(self.arm / "submit_arm.sh"), *arguments],
                              env=self.env, text=True, capture_output=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def record(self):
        return json.loads((self.arm / "submission/submission.json").read_text())

    def test_four_stage_resources_dependencies_and_cpu_plan(self):
        result = self.submit()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 6)
        self.assertTrue(all("--test-only" in call for call in calls[:2]))
        for call in calls:
            for option in ("--account=biup-dtai-gh", "--cpus-per-task=8", "--kill-on-invalid-dep=yes"):
                self.assertIn(option, call)
        for index in (0, 2):
            self.assertIn("--partition=ghx4-interactive", calls[index])
            self.assertNotIn("--partition=ghx4", calls[index])
        for index in (1, 3, 4, 5):
            self.assertIn("--partition=ghx4", calls[index])
            self.assertNotIn("--partition=ghx4-interactive", calls[index])
        self.assertIn("--gpus-per-node=1", calls[2])
        self.assertIn("--mem=48G", calls[2])
        self.assertIn("--time=02:00:00", calls[2])
        for call in calls[3:]:
            self.assertIn("--gpus-per-node=1", call)
            self.assertIn("--mem=32G", call)
        jobs = self.record()["jobs"]
        self.assertEqual(set(jobs), {"build_environment", "validate_first_chunk", "remaining_chunks", "finalize"})
        self.assertEqual(self.record()["gpus_per_stage"], dict.fromkeys(jobs, 1))
        self.assertEqual(self.record()["partitions_per_stage"], {
            "build_environment": "ghx4-interactive", "validate_first_chunk": "ghx4",
            "remaining_chunks": "ghx4", "finalize": "ghx4"})
        self.assertIn("--dependency=afterok:" + jobs["build_environment"], calls[3])
        self.assertIn("--dependency=afterok:" + jobs["validate_first_chunk"], calls[4])
        self.assertIn("--array=1-24%1", calls[4])
        self.assertIn("--dependency=afterok:" + jobs["remaining_chunks"], calls[5])
        self.assertIn("--time=00:30:00", calls[5])
        plan = json.loads((self.arm / "submission/checked_plan.json").read_text())
        self.assertEqual(plan["runtime_profile"], "deltaai-arm-llvm20")
        self.assertFalse((self.arm / "validated_env.json").exists())
        self.assertFalse((self.arm / "submission/in-progress.json").exists())
        self.assertEqual(self.submit().returncode, 0)
        self.assertEqual(len(self.calls()), 6)

    def test_failed_pipeline_recovery_preserves_history_and_rebuilds_chain(self):
        self.assertEqual(self.submit().returncode, 0)
        previous = self.record()
        result = self.submit("--resume-failed")
        self.assertEqual(result.returncode, 0, result.stderr)
        current = self.record()
        self.assertNotEqual(current["jobs"], previous["jobs"])
        self.assertEqual(current["output_dir"], previous["output_dir"])
        self.assertEqual(current["arm_dir"], previous["arm_dir"])
        history = list((self.arm / "submission/attempts").glob("*.json"))
        self.assertEqual(len(history), 1)
        self.assertEqual(json.loads(history[0].read_text()), previous)
        self.assertEqual(Path(current["previous_record"]), history[0])
        self.assertEqual(len(self.calls()), 12)

    def test_running_job_prevents_recovery(self):
        self.assertEqual(self.submit().returncode, 0)
        self.env["LBM500K_FAKE_ACTIVE_JOBS"] = self.record()["jobs"]["build_environment"]
        result = self.submit("--resume-failed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("still active or pending", result.stderr)
        self.assertEqual(len(self.calls()), 6)

    def test_accounting_must_confirm_every_job_terminated(self):
        self.assertEqual(self.submit().returncode, 0)
        self.env["LBM500K_FAKE_ACCOUNTING_STATE"] = "RUNNING"
        self.assertNotEqual(self.submit("--resume-failed").returncode, 0)
        self.env["LBM500K_FAKE_ACCOUNTING_MISSING"] = "1"
        self.assertNotEqual(self.submit("--resume-failed").returncode, 0)
        self.assertEqual(len(self.calls()), 6)

    def test_partial_submission_and_lost_response_prevent_automatic_retry(self):
        self.env["LBM500K_FAKE_FAIL_STAGE"] = "validate_gpu.slurm"
        result = self.submit()
        self.assertNotEqual(result.returncode, 0)
        partial = json.loads((self.arm / "submission/partial.json").read_text())
        self.assertTrue(partial["jobs"]["build_environment"])
        self.assertFalse(partial["jobs"]["validate_first_chunk"])
        count = len(self.calls())
        for arguments in ((), ("--resume-failed",)):
            self.assertNotEqual(self.submit(*arguments).returncode, 0)
        self.assertEqual(len(self.calls()), count)

    def test_in_progress_guard_exists_before_first_job_response(self):
        self.env["LBM500K_FAKE_BAD_ID_STAGE"] = "build.slurm"
        self.assertNotEqual(self.submit().returncode, 0)
        partial = json.loads((self.arm / "submission/partial.json").read_text())
        self.assertTrue((self.arm / "submission/in-progress.json").is_file())
        self.assertFalse(any(partial["jobs"].values()))
        count = len(self.calls())
        self.assertNotEqual(self.submit("--resume-failed").returncode, 0)
        self.assertEqual(len(self.calls()), count)

    def test_wrong_architecture_rejected_without_scheduler_access(self):
        self.env["LBM500K_FAKE_MACHINE"] = "x86_64"
        self.assertNotEqual(self.submit().returncode, 0)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
