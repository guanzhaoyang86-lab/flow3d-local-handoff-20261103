"""Exercise the submission dependency chain using a fake scheduler, never Slurm."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lbm500k_submit_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "package"
        self.package.mkdir()
        for name in ("submit_delta.sh", "generate.slurm", "finalize.slurm", "job_env.sh"):
            shutil.copy2(ROOT / name, self.package / name)
        for name in ("run_generation.py", "source", "upstream"):
            (self.package / name).symlink_to(ROOT / name, target_is_directory=name != "run_generation.py")
        binary = self.root / "bin"
        binary.mkdir()
        (binary / "uname").write_text('#!/bin/bash\nprintf "%s\\n" "${LBM500K_FAKE_MACHINE:-x86_64}"\n')
        fake_sbatch = '''import json,os,pathlib,sys
path=pathlib.Path(os.environ['LBM500K_FAKE_LOG'])
calls=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
args=sys.argv[1:]
calls.append(args)
with path.open('a') as stream:stream.write(json.dumps(args)+'\\n')
if os.environ.get('LBM500K_FAKE_FAIL_ARRAY')=='1' and '--array=1-24%1' in args:
    raise SystemExit(1)
if '--test-only' not in args:print(900000+len(calls))
'''
        (binary / "sbatch").write_text(f"#!{sys.executable}\n" + fake_sbatch)
        fake_squeue = '''import os
print(os.environ.get('LBM500K_FAKE_ACTIVE_JOBS',''))
'''
        (binary / "squeue").write_text(f"#!{sys.executable}\n" + fake_squeue)
        fake_sacct = '''import json,os,pathlib,sys
record=pathlib.Path(os.environ['LBM500K_FAKE_RECORD'])
jobs=json.loads(record.read_text())['jobs']
state=os.environ.get('LBM500K_FAKE_ACCOUNTING_STATE','FAILED')
if os.environ.get('LBM500K_FAKE_ACCOUNTING_MISSING')=='1':
    raise SystemExit(0)
print(jobs['first_chunk']+'|COMPLETED|')
for index in range(1,25):
    print(jobs['remaining_chunks']+'_'+str(index)+'|'+(state if index==1 else 'COMPLETED')+'|')
print(jobs['finalize']+'|CANCELLED|')
'''
        (binary / "sacct").write_text(f"#!{sys.executable}\n" + fake_sacct)
        for path in binary.iterdir():
            path.chmod(0o755)
        self.log = self.root / "scheduler_calls.jsonl"
        prefix = self.root / "env"
        (prefix / "bin").mkdir(parents=True)
        (prefix / "bin/python").symlink_to(sys.executable)
        self.env = os.environ.copy()
        self.env.update(PATH=str(binary) + os.pathsep + self.env["PATH"],
                        LBM500K_FAKE_LOG=str(self.log),
                        LBM500K_FAKE_RECORD=str(self.package / "submission/submission.json"),
                        LBM500K_ENV_PREFIX=str(prefix),
                        LBM500K_OUTPUT_DIR=str(self.root / "dataset"))

    def run_submit(self, *arguments):
        return subprocess.run(["bash", str(self.package / "submit_delta.sh"), *arguments],
                              env=self.env, text=True, capture_output=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_dependencies_and_duplicate_submission_guard(self):
        result = self.run_submit()
        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = json.loads((self.package / "submission/submission.json").read_text())["jobs"]
        calls = self.calls()
        self.assertEqual(len(calls), 4)
        self.assertIn("--test-only", calls[0])
        self.assertIn("--array=1-24%1", calls[2])
        self.assertIn("--dependency=afterok:" + jobs["first_chunk"], calls[2])
        self.assertIn("--dependency=afterok:" + jobs["remaining_chunks"], calls[3])
        self.assertTrue(all("--kill-on-invalid-dep=yes" in call for call in calls))
        self.assertFalse((self.package / "submission/in-progress.json").exists())
        result = self.run_submit()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 4)

    def test_partial_submission_is_preserved_and_cannot_duplicate(self):
        self.env["LBM500K_FAKE_FAIL_ARRAY"] = "1"
        self.assertNotEqual(self.run_submit().returncode, 0)
        record = self.package / "submission/partial.json"
        self.assertTrue(record.is_file())
        self.assertTrue(json.loads(record.read_text())["jobs"]["first_chunk"])
        self.assertTrue((self.package / "submission/in-progress.json").is_file())
        previous = len(self.calls())
        self.assertNotEqual(self.run_submit().returncode, 0)
        self.assertNotEqual(self.run_submit("--resume-failed").returncode, 0)
        self.assertEqual(len(self.calls()), previous)

    def test_failed_pipeline_can_resume_with_new_chain_and_preserved_history(self):
        result = self.run_submit()
        self.assertEqual(result.returncode, 0, result.stderr)
        record_path = self.package / "submission/submission.json"
        previous = json.loads(record_path.read_text())
        result = self.run_submit("--resume-failed")
        self.assertEqual(result.returncode, 0, result.stderr)
        current = json.loads(record_path.read_text())
        self.assertNotEqual(current["jobs"], previous["jobs"])
        self.assertEqual(current["output_dir"], previous["output_dir"])
        self.assertEqual(current["env_prefix"], previous["env_prefix"])
        history = list((self.package / "submission/attempts").glob("*.json"))
        self.assertEqual(len(history), 1)
        self.assertEqual(json.loads(history[0].read_text()), previous)
        self.assertEqual(Path(current["previous_record"]), history[0])
        self.assertEqual(len(self.calls()), 8)
        self.assertIn("--dependency=afterok:" + current["jobs"]["first_chunk"], self.calls()[6])
        self.assertIn("--dependency=afterok:" + current["jobs"]["remaining_chunks"], self.calls()[7])

    def test_resume_refuses_running_or_pending_jobs_before_any_submission(self):
        self.assertEqual(self.run_submit().returncode, 0)
        jobs = json.loads((self.package / "submission/submission.json").read_text())["jobs"]
        self.env["LBM500K_FAKE_ACTIVE_JOBS"] = jobs["remaining_chunks"]
        result = self.run_submit("--resume-failed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("still active or pending", result.stderr)
        self.assertEqual(len(self.calls()), 4)
        self.assertFalse((self.package / "submission/attempts").exists())

    def test_resume_refuses_nonterminal_or_missing_accounting(self):
        self.assertEqual(self.run_submit().returncode, 0)
        self.env["LBM500K_FAKE_ACCOUNTING_STATE"] = "RUNNING"
        result = self.run_submit("--resume-failed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nonterminal", result.stderr)
        self.env["LBM500K_FAKE_ACCOUNTING_MISSING"] = "1"
        result = self.run_submit("--resume-failed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not yet confirm", result.stderr)
        self.assertEqual(len(self.calls()), 4)

    def test_in_progress_without_record_refuses_every_automatic_retry(self):
        directory = self.package / "submission"
        directory.mkdir()
        (directory / "in-progress.json").write_text('{}\n')
        for arguments in ((), ("--resume-failed",)):
            result = self.run_submit(*arguments)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Automatic resubmission is refused", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_resume_requires_original_output_and_incomplete_dataset(self):
        self.assertEqual(self.run_submit().returncode, 0)
        original_output = self.env["LBM500K_OUTPUT_DIR"]
        self.env["LBM500K_OUTPUT_DIR"] = str(self.root / "different-dataset")
        result = self.run_submit("--resume-failed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("original output directory", result.stderr)
        self.env["LBM500K_OUTPUT_DIR"] = original_output
        output = Path(original_output)
        output.mkdir()
        (output / "COMPLETE.json").write_text('{}\n')
        result = self.run_submit("--resume-failed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already complete", result.stderr)
        self.assertEqual(len(self.calls()), 4)

    def test_arm_host_is_rejected_before_scheduler_access(self):
        self.env["LBM500K_FAKE_MACHINE"] = "aarch64"
        result = self.run_submit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DeltaAI", result.stderr)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
