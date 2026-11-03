"""Receiver-side transfer safety checks; all subprocess execution is mocked."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import transfer_assets as transfer


class TransferTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.destination = self.root / 'handoff_20261103'
        self.asset = {
            'id': 'checkpoint-fs_sph-cnn_s31', 'kind': 'file', 'size_bytes': 10,
            'source_path': '/server/results/fs_sph/cnn_s31/latest.pt',
            'destination': 'results/current_baselines/fs_sph/cnn_s31/latest.pt',
        }
        self.save_manifest([self.asset])
        here = mock.patch.object(transfer, 'HERE', self.root)
        here.start()
        self.addCleanup(here.stop)

    def save_manifest(self, assets):
        (self.root / 'asset_manifest.json').write_text(json.dumps({'assets': assets}))

    def arguments(self):
        return ['--asset', self.asset['id'], '--host', 'deltaai',
                '--destination-root', str(self.destination)]

    def test_default_prints_without_connecting_or_creating_target(self):
        output = io.StringIO()
        with mock.patch.object(transfer.subprocess, 'run') as run, \
                mock.patch.object(transfer.shutil, 'which') as which, redirect_stdout(output):
            transfer.main(self.arguments())
        run.assert_not_called()
        which.assert_not_called()
        self.assertFalse(self.destination.exists())
        self.assertIn('rsync -a --partial --protect-args --', output.getvalue())
        self.assertIn('Print only', output.getvalue())

    def test_list_needs_no_host_or_destination_and_does_not_connect(self):
        output = io.StringIO()
        with mock.patch.object(transfer.subprocess, 'run') as run, redirect_stdout(output):
            transfer.main(['--list'])
        run.assert_not_called()
        self.assertIn(self.asset['id'], output.getvalue())
        self.assertIn(self.asset['destination'], output.getvalue())
        self.assertFalse(self.destination.exists())

    def test_execute_pulls_original_filename_to_launcher_path(self):
        with mock.patch.object(transfer.subprocess, 'run') as run, \
                mock.patch.object(transfer.shutil, 'which', return_value='/bin/rsync'), \
                redirect_stdout(io.StringIO()):
            transfer.main(self.arguments() + ['--execute'])
        expected = self.destination / self.asset['destination']
        run.assert_called_once_with([
            'rsync', '-a', '--partial', '--protect-args', '--',
            'deltaai:' + self.asset['source_path'], str(expected)], check=True)
        self.assertTrue(expected.parent.is_dir())
        self.assertFalse(expected.exists())  # A mock, never a real transfer.
        self.assertFalse(any(argument.startswith('--delete') for argument in run.call_args.args[0]))

    def test_directory_copies_contents_without_deletion(self):
        asset = {'id': 'data-fs_sph', 'kind': 'directory', 'source_path': '/server/data/fs_sph',
                 'destination': 'data/fs_sph', 'size_bytes': None}
        self.save_manifest([asset])
        with mock.patch.object(transfer.subprocess, 'run') as run, \
                mock.patch.object(transfer.shutil, 'which', return_value='/bin/rsync'), \
                redirect_stdout(io.StringIO()):
            transfer.main(['--asset', 'data-fs_sph', '--host', 'deltaai',
                           '--destination-root', str(self.destination), '--execute'])
        command = run.call_args.args[0]
        self.assertEqual(command[-2], 'deltaai:/server/data/fs_sph/')
        self.assertEqual(command[-1], str(self.destination / 'data/fs_sph') + '/')
        self.assertFalse(any(argument.startswith('--delete') for argument in command))

    def test_invalid_host_is_rejected_before_any_process(self):
        for host in ('', '-alias', 'delta ai', 'bad\nhost', 'bad\x01host', 'alias;echo', 'alias\x7f'):
            with self.subTest(host=host), mock.patch.object(transfer.subprocess, 'run') as run, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                transfer.main(['--asset', self.asset['id'], '--host=' + host,
                               '--destination-root', str(self.destination), '--execute'])
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_invalid_assets_are_rejected_before_destination_creation(self):
        for changes in ({'destination': '../outside/latest.pt'},
                        {'destination': '/outside/latest.pt'},
                        {'destination': 'results/renamed.pt'},
                        {'source_path': '/server/bad\npath/latest.pt'},
                        {'kind': 'delete'}):
            self.save_manifest([dict(self.asset, **changes)])
            with self.subTest(changes=changes), mock.patch.object(transfer.subprocess, 'run') as run, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                transfer.main(self.arguments() + ['--execute'])
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
            self.assertFalse(self.destination.exists())

    def test_unknown_later_asset_prevents_all_transfers(self):
        with mock.patch.object(transfer.subprocess, 'run') as run, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            transfer.main(self.arguments() + ['--asset', 'not-listed', '--execute'])
        run.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_symlink_escape_is_rejected(self):
        self.destination.mkdir()
        outside = self.root / 'outside'
        outside.mkdir()
        (self.destination / 'results').symlink_to(outside, target_is_directory=True)
        with mock.patch.object(transfer.subprocess, 'run') as run, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            transfer.main(self.arguments() + ['--execute'])
        run.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
