"""Assignment and calibration contracts; no scheduler, GPU, or data download."""
import contextlib
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import local_baselines as base
import local_calibration as calibration


class AssignmentAndCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = base.ROOT
        base.ROOT = Path(self.tmp.name)
        (base.ROOT/'results').mkdir()
        self.queue = dict(captured_at_utc=(datetime.now(timezone.utc)-timedelta(days=4)).isoformat(),jobs=[])
        base.write(base.ROOT/'results/server_queue.json',self.queue)
        base.write(base.ROOT/'results/work_assignment.json',dict(local=dict(datasets=['jhtdb','fs_sph'],
          models=['deeponet','pod','rbf'],stages=['train','calibration']),released_server_jobs=[dict(job_id='123',final_state='CANCELLED')]))

    def tearDown(self):
        base.ROOT = self.old
        self.tmp.cleanup()

    def test_explicit_assignment_survives_old_queue_but_active_job_blocks(self):
        base.assert_training_unassigned('jhtdb','deeponet')
        self.queue['jobs']=[dict(job_id='124',dataset='jhtdb',model='deeponet',stage='train',state='PENDING')]
        base.write(base.ROOT/'results/server_queue.json',self.queue)
        with self.assertRaisesRegex(ValueError,'assigned to the server'):
            base.assert_training_unassigned('jhtdb','deeponet')

    def test_accounted_cancellation_resolves_history_even_after_refresh(self):
        self.queue['previously_assigned_jobs']=[dict(job_id='123',dataset='jhtdb',model='deeponet',stage='train',state='PENDING')]
        base.write(base.ROOT/'results/server_queue.json',self.queue)
        with patch.object(base.subprocess,'run',return_value=SimpleNamespace(stdout='')):
            with contextlib.redirect_stdout(io.StringIO()):
                base.refresh_queue(SimpleNamespace(host='deltaai'))
        base.assert_training_unassigned('jhtdb','deeponet')

    def test_completed_pod_allows_calibration_but_not_refit(self):
        path=base.result_dir('jhtdb','pod');path.mkdir(parents=True)
        base.write(path/'run.json',dict(status='completed'))
        base.assert_training_unassigned('jhtdb','pod',stage='calibration')
        with self.assertRaisesRegex(ValueError,'already has a completed'):
            base.assert_training_unassigned('jhtdb','pod')

    def test_own_model_queue_names(self):
        rows=base.parse_queue_output('42|own3_fs_sph_tensor_dit|PENDING|Dependency|N/A\n43|own3_jhtdb_dps_evaluate|PENDING|Dependency|N/A')
        self.assertEqual((rows[0]['dataset'],rows[0]['model']),('fs_sph','tensor_dit'))
        self.assertEqual(rows[1]['stage'],'evaluate')
        adapter=base.parse_queue_output('44|own3_lbm_adapter|PENDING|Dependency|N/A')[0]
        self.assertEqual((adapter['dataset'],adapter['model'],adapter['stage']),('lbm',None,'prepare'))

    def test_calibration_commands_keep_full_validation_protocol(self):
        for dataset in ('jhtdb','fs_sph'):
            cfg=base.experiment(dataset)/'configs';cfg.mkdir(parents=True)
            base.write(cfg/'rbf_s31.json',dict(fit_steps=1000,observation_seed=47,sigma=.05))
            args=SimpleNamespace(dataset=dataset,model='rbf',device='cuda')
            rows=calibration.commands(args,base.ROOT/'out',dict(manifest='manifest.json',split_counts=dict(validation=18)),None)
            self.assertEqual(len(rows),24)
            for row in rows:
                command=row['command']
                for flag,value in [('--split','validation'),('--seed','31'),('--num-particles','12')]:
                    self.assertEqual(command[command.index(flag)+1],value)
                self.assertEqual(row['config']['fit_steps'],1000)
                self.assertNotIn('--checkpoint',command)
            self.assertFalse((base.ROOT/'out').exists())

    def test_selection_requires_complete_grid_and_deterministic_tie(self):
        rows=[dict(sigma=s,case_index=i,validation_relative_l2=.5) for s in calibration.SIGMAS for i in range(8)]
        self.assertEqual(calibration.select(rows,8)[0],.02)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            calibration.select(rows[:-1],8)

    def test_orphan_prediction_is_never_overwritten(self):
        output=base.ROOT/'case'
        output.with_suffix('.npz').write_bytes(b'partial')
        args=SimpleNamespace(dataset='jhtdb',model='rbf',resume=True)
        with self.assertRaisesRegex(ValueError,'no overwrite'):
            calibration.resume_command(args,dict(output=str(output)),{},None)


if __name__=='__main__':
    unittest.main()
