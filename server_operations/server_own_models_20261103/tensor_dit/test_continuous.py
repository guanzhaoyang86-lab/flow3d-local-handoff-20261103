import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import continuous


class ContinuousGateTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'python': '/pinned/python', 'frozen_root': '/frozen',
                     'datasets': {'jhtdb': {'config': 'configs/jhtdb_s31.json', 'output_dir': '/new/results/jhtdb'}}}
        self.runtime = Mock()
        self.runtime.artifact_ready.return_value = Path('/new/results/jhtdb/tensor_space/selected.pt')
        self.runtime.train_command.return_value = ['formal-train']
        self.runtime.completed_checkpoint.return_value = {'epoch': 100}
        self.updates = []

    def run_stages(self, completed=False):
        with patch.object(continuous, 'data_ready', return_value=Path('/data/manifest.json')), \
             patch.object(continuous, 'completed_formal', return_value=completed), \
             patch.object(continuous, 'sha', return_value='verified-checkpoint-sha'):
            return continuous.run_stages(self.plan, 'jhtdb', self.runtime, lambda a, b: self.updates.append((a, b)))

    def test_success_preserves_stage_order_and_only_seed31(self):
        result = self.run_stages()
        self.assertEqual(result['seed'], 31)
        self.assertEqual(result['epoch'], 100)
        self.assertFalse(result['test_evaluated'])
        self.assertEqual(self.runtime.train_command.call_args.args[-2:], (31, 100))
        self.assertEqual(self.updates, [(s, state) for s in ('tensor_prepare', 'gpu_pilot', 'train_seed31') for state in ('running', 'completed')])
        command = self.runtime.run.call_args_list[0].args[0]
        self.assertIn('--resume', command)
        self.assertEqual(command[command.index('--ranks') + 1], '4,8,12,16')

    def test_failed_rank_gate_blocks_pilot_and_training(self):
        self.runtime.artifact_ready.side_effect = RuntimeError('rank gate failed')
        with self.assertRaisesRegex(RuntimeError, 'rank gate failed'):
            self.run_stages()
        self.runtime.run_pilot.assert_not_called()
        self.runtime.train_command.assert_not_called()

    def test_failed_pilot_blocks_formal_training(self):
        self.runtime.pilot_ready.side_effect = RuntimeError('pilot failed')
        with self.assertRaisesRegex(RuntimeError, 'pilot failed'):
            self.run_stages()
        self.runtime.train_command.assert_not_called()

    def test_completed_formal_is_verified_without_retraining(self):
        self.run_stages(completed=True)
        self.runtime.train_command.assert_not_called()
        self.runtime.completed_checkpoint.assert_called_once()
        self.runtime.formal_scope.assert_called_once()

    def test_preparation_failure_stops_before_artifact_gate(self):
        self.runtime.run.side_effect = RuntimeError('prepare interrupted')
        with self.assertRaisesRegex(RuntimeError, 'prepare interrupted'):
            self.run_stages()
        self.runtime.artifact_ready.assert_not_called()
        self.runtime.run_pilot.assert_not_called()


if __name__ == '__main__':
    unittest.main()
