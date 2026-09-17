"""Run: python -m unittest discover -s test -p test_compute_acc_task_groups.py."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import _bootstrap  # noqa: F401
import mpn_tasks
import net_helpers


class ComputeAccTaskGroupsTests(unittest.TestCase):
    def make_batch(self, *, fixoff=True, padded=True, rule_start=None):
        offset = (6 if fixoff else 5) if rule_start is None else rule_start
        task_ids = torch.tensor([2, 0, 1])
        ends = (16, 18, 20) if padded else (20, 20, 20)
        inputs = torch.zeros(3, 20, offset + 3)
        targets = torch.zeros(3, 20, 3)
        targets[:, :, 2] = 1
        outputs = targets.clone()
        masks = torch.zeros_like(targets)
        for trial_index, end in enumerate(ends):
            inputs[trial_index, :8, 0] = 1
            if fixoff:
                inputs[trial_index, 8:end, 1] = 1
            inputs[trial_index, :end, offset + task_ids[trial_index]] = 1
            masks[trial_index, 2:6] = 1
            masks[trial_index, 8:end] = 1
            if task_ids[trial_index] == 2:
                outputs[trial_index, :, 2] = -1
            elif task_ids[trial_index] == 1:
                outputs[trial_index, 15:end, 2] = -1
        model = SimpleNamespace(loss_type="MSE", prefs=torch.arange(8) * (2 * torch.pi / 8))
        return model, outputs, targets, masks, inputs, task_ids

    def score(self, batch, **kwargs):
        model, outputs, targets, masks, inputs, _ = batch
        return net_helpers.BaseNetwork.compute_acc(
            model, outputs, targets, masks, inputs, mode="angle", **kwargs)

    def assert_matches_task_subsets(self, batch, **kwargs):
        overall, groups = self.score(batch, isvalid=True, **kwargs)
        ungrouped, absent_groups = self.score(batch, isvalid=False)
        self.assertEqual(float(overall), float(ungrouped))
        self.assertIsNone(absent_groups)
        expected = {}
        model, outputs, targets, masks, inputs, task_ids = batch
        for task in torch.unique(task_ids).tolist():
            selected = task_ids == task
            subset_acc, _ = net_helpers.BaseNetwork.compute_acc(
                model, outputs[selected], targets[selected], masks[selected],
                inputs[selected], isvalid=False, mode="angle")
            expected[task] = float(subset_acc)
        self.assertEqual(groups, expected)
        return overall, groups

    def test_padded_and_unpadded_fixation_layouts(self):
        for fixoff in (False, True):
            for padded in (False, True):
                with self.subTest(fixoff=fixoff, padded=padded):
                    batch = self.make_batch(fixoff=fixoff, padded=padded)
                    _, groups = self.assert_matches_task_subsets(batch)
                    self.assertEqual(list(groups), [0, 1, 2])
                    self.assertEqual(groups[0], 1.0)
                    self.assertEqual(groups[2], 0.0)
                    self.assertGreater(groups[1], 0.0)
                    self.assertLess(groups[1], 1.0)

    def test_first_timepoint_alone_is_ambiguous_without_fixoff(self):
        batch = self.make_batch(fixoff=False)
        inputs = batch[4]
        self.assertTrue(torch.all(inputs[:, 0, 0] + inputs[:, 0, 1] == 1))
        self.assert_matches_task_subsets(batch)

    def test_all_fifteen_task_ids_with_shuffled_trials(self):
        model, outputs, targets, masks, base_inputs, _ = self.make_batch()
        task_ids = torch.randperm(15, generator=torch.Generator().manual_seed(17))
        inputs = torch.zeros(15, 20, 21)
        inputs[:, :, :6] = base_inputs[:, :, :6].repeat(5, 1, 1)
        cue_active = base_inputs[:, :, 6:].sum(dim=-1).repeat(5, 1)
        for trial_index, task_id in enumerate(task_ids.tolist()):
            inputs[trial_index, :, 6 + task_id] = cue_active[trial_index]
        batch = (model, outputs.repeat(5, 1, 1), targets.repeat(5, 1, 1),
                 masks.repeat(5, 1, 1), inputs, task_ids)
        _, groups = self.assert_matches_task_subsets(batch)
        self.assertEqual(list(groups), list(range(15)))
        self.assert_matches_task_subsets(batch, rule_start=6)

    def test_explicit_metadata_bypasses_inference(self):
        batch = self.make_batch(rule_start=9)
        batch[4][:, :, :2] = 0
        self.assert_matches_task_subsets(batch, rule_start=9)
        self.assert_matches_task_subsets(batch, rule_start=np.int64(9))

    def test_task_subset_preserves_noncontiguous_ids(self):
        batch = self.make_batch()
        subset = (batch[0], *(tensor[:2] for tensor in batch[1:]))
        _, groups = self.assert_matches_task_subsets(subset)
        self.assertEqual(list(groups), [0, 2])

    def test_invalid_offsets_and_empty_inputs(self):
        batch = self.make_batch()
        for offset in (-1, batch[4].shape[-1], 1.5, True, "6"):
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "rule_start"):
                self.score(batch, isvalid=True, rule_start=offset)
        baseline, _ = self.score(batch, isvalid=False)
        ignored, _ = self.score(batch, isvalid=False, rule_start=-1)
        self.assertEqual(float(baseline), float(ignored))
        batch[4].zero_()
        with self.assertRaisesRegex(ValueError, "all-zero"):
            self.score(batch, isvalid=True)

    def test_uniform_old_offset_preserves_sampling_vector(self):
        batch = self.make_batch()
        original_argidx = net_helpers.one_hot_argidx
        old_running = np.zeros(3)
        new_running = np.zeros(3)
        generator = torch.Generator().manual_seed(42)
        for noise_scale in (0.3, 0.8, 1.3):
            outputs = batch[2] + noise_scale * torch.randn(batch[2].shape, generator=generator)
            noisy_batch = (batch[0], outputs, *batch[2:])
            new_overall, new_groups = self.score(noisy_batch, isvalid=True)
            with patch.object(net_helpers, "one_hot_argidx",
                              side_effect=lambda cues: original_argidx(cues) + 1):
                old_overall, old_groups = self.score(noisy_batch, isvalid=True)
            self.assertEqual(list(old_groups), [1, 2, 3])
            self.assertEqual(float(old_overall), float(new_overall))
            old_values = np.array(list(old_groups.values()))
            new_values = np.array(list(new_groups.values()))
            np.testing.assert_array_equal(old_values, new_values)
            old_running = 0.9 * (old_running + mpn_tasks.normalize_to_one(1 - old_values))
            new_running = 0.9 * (new_running + mpn_tasks.normalize_to_one(1 - new_values))
            np.testing.assert_array_equal(mpn_tasks.normalize_to_one(old_running),
                                          mpn_tasks.normalize_to_one(new_running))


if __name__ == "__main__":
    unittest.main()
