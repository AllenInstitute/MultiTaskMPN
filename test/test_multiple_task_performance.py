"""Regression tests for metadata-aligned task accuracy without model inference."""

import json
from types import SimpleNamespace
import unittest

import torch

import _bootstrap  # noqa: F401
from multiple_task_performance import task_specific_accuracy
from net_helpers import BaseNetwork


class TaskAccuracyTests(unittest.TestCase):
    def batch(self, rule_start=6):
        inputs = torch.zeros(3, 20, rule_start + 3)
        inputs[:, :16, 0] = 1
        inputs[0, :16, rule_start + 2] = 1
        inputs[1, :16, rule_start] = 1
        inputs[2, :16, rule_start + 1] = 1
        targets = torch.zeros(3, 20, 3)
        targets[:, :, 2] = 1
        masks = torch.ones_like(targets)
        masks[:, :2] = 0
        masks[:, 6:8] = 0
        masks[:, 16:] = 0
        outputs = targets.clone()
        outputs[0, :, 2] = -1
        outputs[2, 13:16, 2] = -1
        params = {"rules": ["fdgo", "fdanti", "reactgo"], "hp": {"rule_start": rule_start}}
        model = SimpleNamespace(loss_type="MSE", prefs=torch.arange(8) * (2 * torch.pi / 8))
        model.compute_acc = lambda *args, **kwargs: BaseNetwork.compute_acc(model, *args, **kwargs)
        return model, outputs, targets, masks, inputs, params

    def test_padded_shuffled_trials_and_both_input_layouts(self):
        for rule_start in (5, 6):
            with self.subTest(rule_start=rule_start):
                model, outputs, targets, masks, inputs, params = self.batch(rule_start)
                original = outputs.clone()
                overall, _ = model.compute_acc(outputs, targets, masks, inputs, isvalid=True)
                scores = task_specific_accuracy(model, outputs, targets, masks, inputs, params)
                self.assertEqual(scores, {"fdgo": 1.0, "fdanti": 0.5, "reactgo": 0.0})
                after, _ = model.compute_acc(outputs, targets, masks, inputs, isvalid=True)
                self.assertEqual(float(overall), float(after))
                self.assertEqual(float(overall), 0.5)
                torch.testing.assert_close(outputs, original)
                json.dumps(scores, allow_nan=False)

    def test_absent_task_is_null(self):
        model, outputs, targets, masks, inputs, params = self.batch()
        scores = task_specific_accuracy(model, outputs[:2], targets[:2], masks[:2], inputs[:2], params)
        self.assertEqual(scores, {"fdgo": 1.0, "fdanti": None, "reactgo": 0.0})

    def test_bad_metadata_and_missing_cues_are_rejected(self):
        model, outputs, targets, masks, inputs, params = self.batch()
        params["hp"]["rule_start"] = 5
        with self.assertRaisesRegex(ValueError, "metadata"):
            task_specific_accuracy(model, outputs, targets, masks, inputs, params)
        params["hp"]["rule_start"] = 6
        inputs[0, 0, 6:] = 0
        with self.assertRaisesRegex(ValueError, "active task cue"):
            task_specific_accuracy(model, outputs, targets, masks, inputs, params)


if __name__ == "__main__":
    unittest.main()
