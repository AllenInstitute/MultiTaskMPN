"""Context-length sweep: one directory per length, CLI forwarding and the r vs Context figure."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
import numpy as np
import torch

import paper_plot
import state_space_shift as state_space
import state_space_evaluation as evaluation_utils
from test_state_space_distance_angle import _distance_angle_entry

ANAME = "everything_seed1_L21e4+hidden300+batch128+angle"


class ContextLengthSettingTests(unittest.TestCase):
    def test_steps_round_down_and_the_default_length_keeps_the_default_directory(self):
        self.assertEqual(evaluation_utils.context_steps(480, 40), 12)
        self.assertEqual(evaluation_utils.context_steps(500, 40), 12)
        self.assertEqual(evaluation_utils.context_steps(240, 40), 6)
        self.assertEqual(evaluation_utils.context_steps(10, 40), 1)
        with self.assertRaises(ValueError):
            evaluation_utils.context_steps(0, 40)
        base = Path("state_space")
        self.assertEqual(evaluation_utils.context_dir_of(base, evaluation_utils.FIXED_FIXATION_MS), base)
        self.assertEqual(evaluation_utils.context_dir_of(base, 240.0), base / "context240ms")
        self.assertEqual(evaluation_utils.context_dir_of(base, 960), base / "context960ms")

    def test_cli_puts_another_length_in_its_own_directory_and_forwards_it(self):
        args = state_space._parse_args(["--context-ms", "240"])
        self.assertEqual(Path(args.out_dir), state_space.STATE_SPACE_DIR / "context240ms")
        args = state_space._parse_args(["--context-ms", "240", "--out-dir", "scratch"])
        self.assertEqual(args.out_dir, "scratch")
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            state_space._parse_args(["--context-ms", "0"])
        with patch.object(state_space, "run_all") as run_all, \
                patch.object(state_space, "run_untrained") as run_untrained, \
                patch.object(state_space, "summarize"):
            state_space.main(["--context-ms", "240", "--steps", "5", "--skip-pca"])
            state_space.main(["--context-ms", "960", "--untrained", "--checkpoint", "a.pt"])
        run_all.assert_called_once_with(state_space.STATE_SPACE_DIR / "context240ms",
                                        n_steps=5, skip_pca=True, context_ms=240.0)
        run_untrained.assert_called_once_with(
            ["a.pt"], state_space.STATE_SPACE_DIR / "context960ms", [state_space.UNTRAINED_INIT_SEED],
            n_steps=1, skip_pca=False, context_ms=960.0)

    def test_context_length_reaches_the_evaluation_trials(self):
        with patch.object(evaluation_utils, "prepare_evaluation",
                          side_effect=RuntimeError("stop")) as prepare:
            with self.assertRaises(RuntimeError):
                state_space._context_states("run", trace_steps=5, context_ms=240)
        self.assertEqual(prepare.call_args.kwargs["context_ms"], 240)
        trials = [SimpleNamespace(epochs={"fix1": (0, 6)})]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / f"param_{ANAME}_param.json").write_text(
                json.dumps({"task_params": {}, "train_params": {}, "net_params": {}}))
            with patch.object(evaluation_utils.torch, "load",
                              return_value={"net_params": {}, "state_dict": {}}), \
                    patch.object(evaluation_utils, "build_network", return_value=(Mock(), {})), \
                    patch.object(evaluation_utils, "modulation_retention", return_value=0.99), \
                    patch.object(evaluation_utils.mpn_tasks, "convert_and_init_multitask_params",
                                 return_value=({"rules": ["fdgo"], "hp": {"dt": 40}}, {}, {})), \
                    patch.object(evaluation_utils.mpn_tasks, "generate_trials_wrap",
                                 return_value=((None, None, None), (None, trials, [0]))) as generate, \
                    patch("builtins.print"):
                evaluation = evaluation_utils.prepare_evaluation(
                    root / f"savednet_{ANAME}.pt", device=torch.device("cpu"), context_ms=240)
        self.assertEqual(generate.call_args.kwargs["fixed_fixation_steps"], 6)
        self.assertEqual(evaluation["fixed_fixation_steps"], 6)


def _results(length, offset=0.0, n=2):
    """Distance-angle results of ``n`` L2 = 1e-4 runs at one Context length,
    their r shifted by ``offset`` from the fixture's."""
    results = {}
    for index in range(n):
        entry = _distance_angle_entry()
        entry.update(fixed_fixation_ms=length, fixed_fixation_steps=int(length / 40))
        entry["rval_dict"] = {name: (r + offset, slope, p)
                              for name, (r, slope, p) in entry["rval_dict"].items()}
        results[f"everything_seed{index}_L21e4+hidden300+batch128+angle"] = entry
    return results


class ContextSweepFigureTests(unittest.TestCase):
    def _write(self, directory, results):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / state_space.RESULTS_NAME).open("wb") as stream:
            pickle.dump(results, stream)

    def _draw(self, root):
        with patch.object(paper_plot, "STATE_SPACE_DIR", root), \
                patch.object(paper_plot, "_ensure_out_dir"), \
                patch.object(paper_plot, "_load_best_state_space_trial",
                             side_effect=AssertionError("PCA ranking used")), \
                patch.object(paper_plot, "_save_fig") as save, patch("builtins.print"):
            paper_plot.plot_state_space_r_vs_context()
        if not save.called:
            return None
        fig = save.call_args.args[0]
        self.addCleanup(paper_plot.plt.close, fig)
        self.assertEqual(save.call_args.args[1].name, "multitask_state_space_r_vs_context.png")
        return fig

    def test_one_line_per_representation_over_the_lengths_found(self):
        source = "everything_seed0_L21e4+hidden300+batch128+angle"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, _results(480.0))
            self._write(root / "context960ms", _results(960.0, offset=0.05))
            self._write(root / "context240ms", _results(240.0, offset=-0.2))
            untrained = {}
            for seed in (0, 1):
                entry = _distance_angle_entry()
                entry.update(untrained=True, init_seed=seed, l2_info=None,
                             fixed_fixation_ms=240.0, config_source=source)
                untrained[f"{source}_init{seed}"] = entry
            self._write(root / "context240ms" / "untrained", untrained)
            fig = self._draw(root)
        self.assertEqual(len(fig.axes), 5)
        trained = fig.axes[2]  # L2 = 1e-4
        self.assertIn("n=2", trained.get_title())
        self.assertEqual(len(trained.lines), len(paper_plot._STATE_SPACE_R_BARS) + 1)  # + zero line
        np.testing.assert_array_equal(trained.lines[0].get_xdata(), [240, 480, 960])
        np.testing.assert_allclose(trained.lines[0].get_ydata(), [0.7, 0.9, 0.95])  # hidden
        np.testing.assert_allclose(trained.lines[4].get_ydata(), [0.3, 0.5, 0.55])  # eff_mod_hebb
        self.assertEqual([text.get_text() for text in trained.get_xticklabels()], ["240", "480", "960"])
        untrained_panel = fig.axes[0]
        self.assertIn("n=0-2", untrained_panel.get_title())
        values = untrained_panel.lines[0].get_ydata()
        self.assertAlmostEqual(values[0], 0.9)
        self.assertTrue(np.isnan(values[1:]).all())

    def test_skipped_below_two_lengths_or_with_mixed_lengths_in_one_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, _results(480.0))
            self.assertIsNone(self._draw(root))
            mixed = {**_results(240.0), **_results(960.0, n=1)}
            self._write(root / "context240ms", mixed)
            self.assertIsNone(self._draw(root))


if __name__ == "__main__":
    unittest.main()
