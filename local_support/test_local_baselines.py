"""CPU-only contract tests; no Slurm calls, downloads, or real training."""
import contextlib
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('local_baselines', Path(__file__).with_name('local_baselines.py'))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class LocalHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_root = module.ROOT
        module.ROOT = Path(self.temp.name)
        (module.ROOT/'results').mkdir()
        self.server = {'captured_at_utc': datetime.now(timezone.utc).isoformat(), 'jobs': []}
        self.write_server()

    def tearDown(self):
        module.ROOT = self.old_root
        self.temp.cleanup()

    def write_server(self):
        module.write(module.ROOT/'results/server_queue.json', self.server)

    def test_pending_server_task_blocks_training_before_any_data_or_gpu_access(self):
        self.server['jobs'] = [{'dataset':'jhtdb', 'model':'cnn', 'stage':'train', 'state':'PENDING', 'job_id':'123'}]
        self.write_server()
        with contextlib.redirect_stderr(io.StringIO()):
            code = module.main(['train','--dataset','jhtdb','--model','cnn','--data-root','not-present','--execute'])
        self.assertEqual(code, 2)
        self.assertFalse((module.ROOT/'local_runs').exists())

    def test_stale_server_snapshot_blocks_formal_training(self):
        self.server['captured_at_utc'] = (datetime.now(timezone.utc)-timedelta(days=2)).isoformat()
        self.write_server()
        with self.assertRaisesRegex(ValueError, 'older than 24 hours'):
            module.assert_training_unassigned('jhtdb','cnn')

    def test_malformed_server_snapshot_fails_closed(self):
        self.server['jobs'] = [{'job_id':'123', 'state':'RUNNING'}]
        self.write_server()
        with self.assertRaisesRegex(ValueError, 'dataset/model/stage/state'):
            module.assert_training_unassigned('jhtdb','cnn')

    def test_completed_imported_result_prevents_retraining(self):
        root = module.result_dir('jhtdb','cnn');root.mkdir(parents=True)
        module.write(root/'run.json',{'status':'completed'})
        with self.assertRaisesRegex(ValueError, 'already has a completed'):
            module.assert_training_unassigned('jhtdb','cnn')

    def test_unassigned_task_allowed_with_fresh_snapshot(self):
        self.assertEqual(module.assert_training_unassigned('jhtdb','cnn'),self.server)

    def test_source_hash_detects_crlf_change(self):
        root = module.experiment('fs_sph');root.mkdir(parents=True)
        path = root/'test.py';path.write_bytes(b'a = 1\n')
        module.write(root/'frozen_manifest.json',{'files':{'test.py':module.sha(path)}})
        self.assertEqual(module.verify_source('fs_sph')['checked_frozen_files'],1)
        path.write_bytes(b'a = 1\r\n')
        with self.assertRaisesRegex(ValueError, 'CRLF'):
            module.verify_source('fs_sph')

    def test_relative_manifest_survives_relocation_and_missing_archive_fails(self):
        root = module.ROOT/'data';root.mkdir()
        rows = {}
        for split in ('train','validation','test'):
            (root/(split+'.npz')).write_bytes(b'fixture')
            rows[split] = [{'case_id':split,'path':split+'.npz'}]
        module.write(root/'manifest.json',{'splits':rows})
        module.write(root/'COMPLETE.json',{'status':'validated','manifest_sha256':module.sha(root/'manifest.json')})
        original = (root/'manifest.json').read_bytes()
        moved = module.ROOT/'new-location';root.rename(moved)
        self.assertEqual(module.verify_data(moved)['split_counts'],{'train':1,'validation':1,'test':1})
        self.assertEqual((moved/'manifest.json').read_bytes(),original)
        (moved/'validation.npz').unlink()
        with self.assertRaisesRegex(ValueError,'archive missing'):
            module.verify_data(moved)

    def test_smoke_commands_only_select_validation_and_separate_predictions(self):
        data = {'manifest':str(module.ROOT/'data/manifest.json')}
        args = types.SimpleNamespace(command='smoke',dataset='jhtdb',model='rbf',device='cpu',case_index=0)
        command = module.build_command(args,module.ROOT/'local_runs/check',data,module.ROOT/'config.json')
        self.assertIn('predict',command)
        self.assertEqual(command[command.index('--split')+1],'validation')
        self.assertNotIn('--checkpoint',command)
        args.dataset = 'fs_sph'
        command = module.build_command(args,module.ROOT/'local_runs/check',data,module.ROOT/'config.json')
        self.assertIn('evaluate',command)
        self.assertEqual(command[command.index('--case-indices')+1],'0')

    def test_queue_parser_maps_training_calibration_and_array_jobs(self):
        jobs = module.parse_queue_output('3492898|b12_jhtdb_deeponet|PENDING|Priority|N/A\n3487248|b12_fs_sph_pod_calibration|PENDING|Dependency|N/A\n3497469_[8-24%1]|lbm500k-arm-part|PENDING|JobArrayTaskLimit|N/A\n')
        self.assertEqual((jobs[0]['dataset'],jobs[0]['model'],jobs[0]['stage']),('jhtdb','deeponet','train'))
        self.assertEqual((jobs[1]['dataset'],jobs[1]['model'],jobs[1]['stage']),('fs_sph','pod','calibration'))
        self.assertEqual(jobs[2]['dataset'],'lbm')

    def test_bad_remote_output_preserves_existing_queue_snapshot(self):
        path = module.ROOT/'results/server_queue.json';original = path.read_bytes()
        with patch.object(module.subprocess,'run',return_value=types.SimpleNamespace(stdout='unexpected login banner')):
            with self.assertRaisesRegex(ValueError,'previous snapshot was preserved'):
                module.refresh_queue(types.SimpleNamespace(host='deltaai'))
        self.assertEqual(path.read_bytes(),original)

    def test_valid_empty_queue_is_accepted_only_after_successful_ssh(self):
        with patch.object(module.subprocess,'run',return_value=types.SimpleNamespace(stdout='')) as mocked:
            with contextlib.redirect_stdout(io.StringIO()):
                module.refresh_queue(types.SimpleNamespace(host='deltaai'))
        self.assertEqual(module.snapshot()['jobs'],[])
        self.assertTrue(mocked.call_args.kwargs['check'])
        self.assertEqual(mocked.call_args.args[0][-1],"squeue -h -u zguan2 -o '%i|%j|%T|%r|%S'")

    def test_disappeared_server_job_remains_blocked_after_refresh(self):
        self.server['jobs'] = [{'job_id':'123','dataset':'jhtdb','model':'cnn','stage':'train','state':'PENDING'}]
        self.write_server()
        with patch.object(module.subprocess,'run',return_value=types.SimpleNamespace(stdout='')):
            with contextlib.redirect_stdout(io.StringIO()):
                module.refresh_queue(types.SimpleNamespace(host='deltaai'))
        with self.assertRaisesRegex(ValueError,'disappeared from squeue'):
            module.assert_training_unassigned('jhtdb','cnn')

    def test_ssh_host_shell_syntax_is_rejected(self):
        with patch.object(module.subprocess,'run') as mocked:
            with self.assertRaisesRegex(ValueError,'SSH config alias'):
                module.refresh_queue(types.SimpleNamespace(host='host; touch file'))
        mocked.assert_not_called()

    def test_train_preserves_seed31_n12_total100(self):
        args = types.SimpleNamespace(command='train',dataset='jhtdb',model='cnn',device='cuda',resume=None)
        command = module.build_command(args,module.ROOT/'local_runs/check',{'manifest':'manifest.json'},module.ROOT/'config.json')
        for flag,value in [('--seed','31'),('--epochs','100'),('--num-particles','12')]:
            self.assertEqual(command[command.index(flag)+1],value)


if __name__ == '__main__':
    unittest.main()
