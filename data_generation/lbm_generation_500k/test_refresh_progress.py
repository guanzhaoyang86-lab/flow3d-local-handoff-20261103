"""Small CPU-only checks for progress accounting after failed ARM builds."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import refresh_progress as progress


class ProgressTest(unittest.TestCase):
    def test_only_validated_chunk_counts_are_used(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/"chunks").mkdir()
            (output/"partial.npz").write_bytes(b"not a valid archive")
            record = dict(status="failed", chunk_index=0, case_start=0, case_stop=125,
                          num_cases=125, generated=2, reused=1, completed_cases=3)
            (output/"chunks/chunk_000.json").write_text(json.dumps(record))
            count, _, errors = progress.validated_chunk_counts(output)
            self.assertEqual(count, 3)
            self.assertEqual(errors, [])
            record["case_start"] = 1
            (output/"chunks/chunk_000.json").write_text(json.dumps(record))
            count, _, errors = progress.validated_chunk_counts(output)
            self.assertEqual(count, 0)
            self.assertEqual(len(errors), 1)

    def test_empty_npz_files_do_not_mean_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/"case.npz").touch()
            submission = {"output_dir": str(output), "arm_dir": str(output),
                          "jobs": dict(zip(progress.STAGES, ("1", "2", "3", "4")))}
            snapshot = {"1": {"state": "FAILED", "exit_code": "1:0"},
                        "3_1": {"state": "CANCELLED", "exit_code": "0:0"}}
            result = progress.refresh({"request": {"keep": True}}, submission, snapshot, [])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["completed_physical_tracks"], 0)
            self.assertEqual(result["request"], {"keep": True})

    def test_queue_overrides_accounting_and_array_failure_is_kept(self):
        jobs = dict(zip(progress.STAGES, ("1", "2", "3", "4")))
        def runner(command, **kwargs):
            output = ("1|COMPLETED|0:0\n3_1|CANCELLED by 10|0:0\n1.batch|FAILED|1:0\n"
                      if command[0] == "sacct" else "2|RUNNING|gh123|2026-10-24T10:00:00\n")
            return subprocess.CompletedProcess(command, 0, output, "")
        snapshot, errors = progress.scheduler_snapshot(jobs, runner)
        self.assertEqual(errors, [])
        self.assertEqual(snapshot["3_1"]["state"], "CANCELLED")
        self.assertEqual(snapshot["2"]["source"], "queue")
        self.assertNotIn("1.batch", snapshot)

    def test_finished_jobs_are_not_requested_from_live_queue(self):
        jobs = dict(zip(progress.STAGES, ("1", "2", "3", "4")))
        def runner(command, **kwargs):
            if command[0] == "sacct":
                return subprocess.CompletedProcess(command, 0,
                    "1|COMPLETED|0:0\n2|COMPLETED|0:0\n3_1|COMPLETED|0:0\n"
                    "3_2|COMPLETED|0:0\n4|COMPLETED|0:0\n", "")
            self.assertIn("--user=" + progress.getpass.getuser(), command)
            self.assertFalse(any(arg.startswith("--jobs=") for arg in command))
            return subprocess.CompletedProcess(command, 0, "", "")
        snapshot, errors = progress.scheduler_snapshot(jobs, runner)
        self.assertEqual(errors, [])
        self.assertEqual(snapshot["4"],
                         {"state": "COMPLETED", "source": "accounting", "exit_code": "0:0"})
        self.assertEqual(snapshot["3_2"]["state"], "COMPLETED")

    def test_mixed_finished_and_live_jobs_filter_unrelated_user_jobs(self):
        jobs = dict(zip(progress.STAGES, ("1", "2", "3", "4")))
        def runner(command, **kwargs):
            if command[0] == "sacct":
                output = "1|COMPLETED|0:0\n2|PENDING|0:0\n3_1|PENDING|0:0\n4|PENDING|0:0\n"
            else:
                self.assertFalse(any(arg.startswith("--jobs=") for arg in command))
                output = ("2|RUNNING|gh123|2026-10-24T10:00:00\n"
                          "3_1|PENDING|(Dependency)|N/A\n"
                          "4|PENDING|(Dependency)|N/A\n"
                          "999|FAILED|unrelated|N/A\n999_1|RUNNING|gh124|N/A\n")
            return subprocess.CompletedProcess(command, 0, output, "")
        snapshot, errors = progress.scheduler_snapshot(jobs, runner)
        self.assertEqual(errors, [])
        self.assertEqual(snapshot["1"]["state"], "COMPLETED")
        self.assertEqual(snapshot["2"]["state"], "RUNNING")
        self.assertEqual(snapshot["2"]["source"], "queue")
        self.assertNotIn("999", snapshot)
        self.assertNotIn("999_1", snapshot)

    def test_real_queue_failure_is_kept_after_terminal_accounting(self):
        jobs = dict(zip(progress.STAGES, ("1", "2", "3", "4")))
        def runner(command, **kwargs):
            if command[0] == "sacct":
                return subprocess.CompletedProcess(command, 0,
                    "1|COMPLETED|0:0\n2|COMPLETED|0:0\n3_1|COMPLETED|0:0\n4|COMPLETED|0:0\n", "")
            raise subprocess.CalledProcessError(1, command, output="",
                stderr="slurm_load_jobs error: Unable to contact slurm controller\n")
        snapshot, errors = progress.scheduler_snapshot(jobs, runner)
        self.assertEqual(snapshot["4"]["state"], "COMPLETED")
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("queue:"))

    def test_queue_errors_are_not_hidden_without_terminal_accounting(self):
        jobs = dict(zip(progress.STAGES, ("1", "2", "3", "4")))
        def runner(command, **kwargs):
            if command[0] == "sacct":
                return subprocess.CompletedProcess(command, 0, "1|RUNNING|0:0\n", "")
            # This error is unexpected for a user-wide query; preserve it.
            raise subprocess.CalledProcessError(1, command, output="",
                stderr="slurm_load_jobs error: Invalid job id specified\n")
        snapshot, errors = progress.scheduler_snapshot(jobs, runner)
        self.assertEqual(snapshot["1"]["state"], "RUNNING")
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("queue:"))

    def test_completion_requires_final_job_success_and_valid_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/"SHA256SUMS").write_text("digest  sample.npz\n")
            (output/"manifest.json").write_text("{}")
            marker = dict(status="completed", num_cases=3125, physical_trajectories=500000,
                          single_view_trajectories=500000, num_independent_flow_groups=3125,
                          projection_names=["xz"], dataset_version=progress.digest(output/"SHA256SUMS"))
            (output/"COMPLETE.json").write_text(json.dumps(marker))
            submission = {"output_dir": str(output), "arm_dir": str(output),
                          "jobs": dict(zip(progress.STAGES, ("1", "2", "3", "4")))}
            failed = progress.refresh({}, submission, {"4": {"state": "FAILED", "exit_code": "1:0"}}, [])
            self.assertEqual(failed["status"], "failed")
            done = progress.refresh({}, submission, {"4": {"state": "COMPLETED", "exit_code": "0:0"}}, [])
            self.assertEqual(done["status"], "completed")
            self.assertEqual(done["completed_physical_tracks"], 500000)
            marker["single_view_trajectories"] = 1500000
            (output/"COMPLETE.json").write_text(json.dumps(marker))
            self.assertFalse(progress.completion_marker(output)[0])

    def test_continuous_job_states_and_completion_ignore_retired_array(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            arm = output / "arm"
            arm.mkdir()
            (arm / "compiler.patch").write_text("test compiler patch\n")
            patch_sha = progress.digest(arm / "compiler.patch")
            (arm / "build_report.json").write_text(json.dumps({
                "status": "compiled_and_installed", "compiler_patch_sha256": patch_sha}))
            (arm / "validated_env.json").write_text(json.dumps({
                "cuda_kernel_validated": True, "lbm32_pilot_validated": True,
                "compiler_patch_sha256": patch_sha,
                "build_report_sha256": progress.digest(arm / "build_report.json")}))
            (output / "chunks").mkdir()
            (output / "chunks/chunk_000.json").write_text(json.dumps({
                "status": "completed", "chunk_index": 0, "case_start": 0,
                "case_stop": 125, "num_cases": 125, "completed_cases": 125,
                "generated": 125, "reused": 0, "slurm_job_id": "2"}))
            submission = {"execution_mode": "continuous", "output_dir": str(output),
                          "arm_dir": str(arm), "superseded_jobs": ["3", "4"],
                          "jobs": dict(zip(progress.STAGES, ("1", "2", "5", "5")))}

            def snapshot(state, exit_code="0:0"):
                def runner(command, **kwargs):
                    if command[0] == "sacct":
                        self.assertIn("--jobs=1,2,5", command)
                        text = ("1|COMPLETED|0:0\n2|COMPLETED|0:0\n"
                                "3_8|CANCELLED by 10|0:0\n4|CANCELLED|0:0\n"
                                f"5|{state}|{exit_code}\n5.batch|FAILED|1:0\n")
                    else:
                        text = (f"5|{state}|Priority|N/A\n"
                                if state in ("PENDING", "RUNNING") else "")
                    return subprocess.CompletedProcess(command, 0, text, "")
                values, errors = progress.scheduler_snapshot(
                    submission["jobs"], runner, execution_mode=submission["execution_mode"])
                self.assertEqual(errors, [])
                self.assertEqual(set(values), {"1", "2", "5"})
                return values

            for state, expected in (("PENDING", "waiting_for_generation_allocation"),
                                    ("RUNNING", "running_generation"),
                                    ("FAILED", "failed"),
                                    ("COMPLETED", "awaiting_scheduler_accounting")):
                with self.subTest(state=state, marker=False):
                    result = progress.refresh({}, submission, snapshot(state), [])
                    self.assertEqual(result["status"], expected)
                    self.assertEqual(result["completed_production_cases"], 125)
                    self.assertEqual(result["completed_physical_tracks"], 20000)
                    self.assertEqual(result["progress_errors"], [])

            (output / "SHA256SUMS").write_text("digest  sample.npz\n")
            (output / "manifest.json").write_text("{}")
            (output / "COMPLETE.json").write_text(json.dumps({
                "status": "completed", "num_cases": 3125,
                "physical_trajectories": 500000, "single_view_trajectories": 500000,
                "num_independent_flow_groups": 3125, "projection_names": ["xz"],
                "dataset_version": progress.digest(output / "SHA256SUMS")}))
            done = progress.refresh({}, submission, snapshot("COMPLETED"), [])
            self.assertEqual(done["status"], "completed")
            self.assertEqual(done["completed_physical_tracks"], 500000)
            for state, exit_code in (("FAILED", "1:0"), ("COMPLETED", "1:0"),
                                     ("RUNNING", "0:0")):
                with self.subTest(state=state, exit_code=exit_code, marker=True):
                    result = progress.refresh({}, submission, snapshot(state, exit_code), [])
                    self.assertNotEqual(result["status"], "completed")

    def test_continuous_alias_requires_explicit_mode_and_three_distinct_ids(self):
        def no_query(*args, **kwargs):
            self.fail("Invalid submission must be rejected before querying Slurm")

        jobs = dict(zip(progress.STAGES, ("1", "2", "3", "3")))
        with self.assertRaises(ValueError):
            progress.scheduler_snapshot(jobs, no_query)
        for ids in (("1", "2", "3", "4"), ("1", "1", "3", "3"),
                    ("1", "2", "1", "1"), ("1", "2", "2", "2"),
                    ("1", "2", "2", "3"), ("1", "1", "1", "1")):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                progress.scheduler_snapshot(dict(zip(progress.STAGES, ids)), no_query,
                                            execution_mode="continuous")
        with self.assertRaises(ValueError):
            progress.scheduler_snapshot(jobs, no_query, execution_mode="unknown")

    def test_continuous_job_ids_remain_strictly_numeric(self):
        def no_query(*args, **kwargs):
            self.fail("Invalid job ID must be rejected before querying Slurm")

        for invalid in ("", "3_1", " 3", "3;4", "-1", 3.5, True, None):
            with self.subTest(job_id=invalid), self.assertRaises(ValueError):
                progress.scheduler_snapshot(
                    dict(zip(progress.STAGES, ("1", "2", invalid, invalid))), no_query,
                    execution_mode="continuous")

    def test_cli_passes_continuous_submission_mode_to_scheduler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission = {"execution_mode": "continuous", "output_dir": str(root),
                          "arm_dir": str(root),
                          "jobs": dict(zip(progress.STAGES, ("1", "2", "5", "5")))}
            path = root / "submission.json"
            path.write_text(json.dumps(submission))
            with mock.patch.object(sys, "argv", ["refresh_progress.py", "--submission", str(path),
                                                  "--progress", str(root / "progress.json")]), \
                    mock.patch.object(progress, "scheduler_snapshot", return_value=({}, [])) as query, \
                    redirect_stdout(io.StringIO()):
                progress.main()
            query.assert_called_once_with(submission["jobs"], execution_mode="continuous")


if __name__ == "__main__":
    unittest.main()
