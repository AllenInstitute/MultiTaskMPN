"""Tests for the chunked recording of the multi-task analysis's trajectories."""

import inspect
import unittest

import _bootstrap  # noqa: F401
import numpy as np
import torch

import multiple_task_analysis
from test_modulation_feedback import _tiny_network


class RecordTrackedStatesTests(unittest.TestCase):
    def test_matches_concatenated_chunks_exactly(self):
        """The preallocated arrays hold what collecting the same chunks and
        concatenating them (the code they replace) held, bit for bit."""
        model = _tiny_network(eta=3.0)
        inputs = torch.randn(7, 9, 4, generator=torch.Generator().manual_seed(0))
        outputs, states = multiple_task_analysis.record_tracked_states(model, inputs, chunk_size=3)
        chunk_outputs, chunk_records = [], []
        with torch.no_grad():
            for start in range(0, 7, 3):
                chunk_output, _, record = model.iterate_sequence_batch(
                    inputs[start:start + 3], run_mode="track_states")
                chunk_outputs.append(chunk_output.cpu())
                chunk_records.append(record)
        torch.testing.assert_close(outputs, torch.cat(chunk_outputs), rtol=0, atol=0)
        self.assertEqual(set(states), set(multiple_task_analysis.TRACKED_STATE_KEYS))
        for key, values in states.items():
            expected = torch.cat([record[key].cpu() for record in chunk_records]).numpy()
            self.assertIsInstance(values, np.ndarray)
            self.assertEqual(values.dtype, expected.dtype)
            np.testing.assert_array_equal(values, expected)
        self.assertEqual(states["M1"].shape, (7, 9, 6, 5))

    def test_the_analysis_holds_no_weighted_copy_of_the_trajectory(self):
        source = inspect.getsource(multiple_task_analysis.main)
        self.assertIn("record_tracked_states(model, test_input, forward_chunk_size)", source)
        self.assertNotIn("Ms_orig * modulation_W", source)
        self.assertNotIn("torch.cat(chunks", source)


if __name__ == "__main__":
    unittest.main()
