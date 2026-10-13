"""The measured PSP-train figure in cartoon.py redraws the extracted trial data like Fig. 4A."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import _bootstrap  # noqa: F401
import cartoon
import paper_plot


WAVEFORM_TIME_MS = np.arange(-1.0, 8.01, 0.5)
INTERVAL_MS = 20.0


def write_full_format(path, n_trials=(4, 3, 5), with_noise=True, with_classes=True):
    """A biology_fast_synaptic_trial_amplitudes.json with all 12 pulses."""
    rng = np.random.RandomState(0)
    connections = []
    classes = {"facilitating": ("L5 IT", "Sst"), "pseudolinear": ("L6 Pyr", "Pv"),
               "depressing": ("L5 ET", "Pv")}
    for group, n in zip(("facilitating", "pseudolinear", "depressing"), n_trials):
        scale = {"facilitating": 1.4, "pseudolinear": 1.0, "depressing": 0.5}[group]
        pulses = [str(p) for p in range(12)]
        entry = {
            "group": group, "title": group.title(), "synapse_type": "excitatory",
            "pair_external_id": ["exp", "1", "2"],
            "n_trials": {p: n for p in pulses},
            "trial_amplitude_mv": {
                p: (rng.rand(n) * (scale ** (int(p) / 7)) + 0.1).tolist() for p in pulses},
            "trial_peak_mv": {p: (rng.rand(n) + 0.5).tolist() for p in pulses},
            "waveform_time_ms": {p: WAVEFORM_TIME_MS.tolist() for p in pulses},
            "mean_waveform_mv": {
                p: (0.6 * scale ** (int(p) / 7) * np.exp(-((WAVEFORM_TIME_MS - 2.0) / 2.0) ** 2)
                    * (WAVEFORM_TIME_MS > 0)).tolist() for p in pulses},
        }
        if with_noise:
            entry["trial_noise_mv"] = {p: (0.05 * rng.randn(n)).tolist() for p in pulses}
        if with_classes:
            entry["pre_class"], entry["post_class"] = classes[group]
        connections.append(entry)
    data = {"protocol": {"frequency_hz": 50.0, "recovery_ms": 250.0},
            "amplitude_statistic": "deconvolution fit amplitude",
            "connections": connections}
    Path(path).write_text(json.dumps(data))
    return data


def write_legacy_format(path):
    """The older waveform metadata: per-trial peaks for five pulses only."""
    labels = ("Pulse 1", "Pulse 4", "Pulse 8", "Pulse 9 (recovery)", "Pulse 12")
    data = {"protocol": {"frequency_hz": 50.0, "recovery_ms": 250.0},
            "connections": [{
                "group": "depressing", "title": "Depressing",
                "pair_external_id": ["exp", "3", "4"],
                "pulses": {label: {"n_trials": 3,
                                   "trial_mean_delta_vm_mv": [0.1, 0.2, 0.3],
                                   "trial_peak_delta_vm_mv": [1.0, 0.8 - i * 0.1, 0.6]}
                           for i, label in enumerate(labels)},
            }]}
    Path(path).write_text(json.dumps(data))
    return data


class SynapticTrainFigureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def render(self, data_path, **kwargs):
        # cartoon imported `_save_fig` by name, so both bindings are patched: the
        # main figure goes through cartoon._save, the colorbar through paper_plot.
        save_fig = Mock()
        with patch.object(cartoon, "_save_fig", save_fig), \
                patch.object(paper_plot, "_save_fig", save_fig):
            cartoon.plot_synaptic_train_amplitudes(out_dir=self.root, data_path=data_path,
                                                   **kwargs)
        for call in save_fig.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        return save_fig

    @staticmethod
    def waveform_lines(axis):
        return [line for line in axis.lines if line.get_color() == "black" and line.get_zorder() == 100]

    @staticmethod
    def break_strokes(axis):
        return [line for line in axis.lines if line.get_zorder() == 200]

    def test_default_draws_selected_pulses_in_slots_labeled_with_real_time(self):
        path = self.root / "amps.json"
        data = write_full_format(path)
        save_fig = self.render(path)
        names = [call.args[1].name for call in save_fig.call_args_list]
        self.assertEqual(names, ["cartoon_synaptic_train_amplitudes.png",
                                 "cartoon_synaptic_train_amplitudes_colorbar.png"])
        extra = save_fig.call_args_list[0].kwargs["extra"]
        self.assertIn("MEASURED", extra)
        self.assertIn("pulses 1 (0 ms), 4 (60 ms), 6 (100 ms), 10 (410 ms), 12 (450 ms)", extra)
        self.assertIn("adjusted log CV on pulse 1 and pulse 6", extra)
        self.assertIn("depressing: L5 ET\u2192Pv", extra)
        figure = save_fig.call_args_list[0].args[0]
        self.assertEqual(len(figure.axes), 3)
        shown = [0, 3, 5, 9, 11]                      # 0-based pulses of SYNAPTIC_PULSES
        by_group = {entry["group"]: entry for entry in data["connections"]}
        # Dots above a row's fixed top are clipped; the message counts them (the
        # three tops differ, so each row's note is identifiable by its top).
        for group, entry in by_group.items():
            top = cartoon.SYNAPTIC_YLIM_TOP[group]
            n_above = sum(int(np.sum(np.asarray(entry["trial_amplitude_mv"][str(p)]) > top))
                          for p in shown)
            if n_above:
                self.assertIn(f"{group} {n_above} dot(s) above {top:g} mV clipped", extra)
            else:
                self.assertNotIn(f"above {top:g} mV clipped", extra)
        grey = paper_plot.mpl.colors.to_rgba(cartoon._SYNAPTIC_DOT_COLOR)
        cmap = paper_plot.plt.get_cmap(cartoon._SYNAPTIC_CV_CMAP)
        norm = paper_plot.mpl.colors.Normalize(*cartoon.SYNAPTIC_VARIABILITY_RANGE, clip=True)
        half_width_slots = cartoon.SYNAPTIC_SWARM_HALF_WIDTH_MS / INTERVAL_MS
        # Paper row order, and no in-axes labels: the caption names the rows.
        for axis, group in zip(figure.axes, ("depressing", "facilitating", "pseudolinear")):
            entry = by_group[group]
            self.assertEqual(len(axis.texts), 0)  # ArtistList, not a plain list
            # One slot per shown pulse: a swarm of one dot per trial centered on
            # the slot, and the pulse's mean waveform at its true latency.
            self.assertEqual(len(axis.collections), len(shown))
            for index, (pulse, collection) in enumerate(zip(shown, axis.collections)):
                offsets = collection.get_offsets()
                np.testing.assert_allclose(
                    sorted(offsets[:, 1]), sorted(entry["trial_amplitude_mv"][str(pulse)]))
                self.assertTrue(np.all(np.abs(offsets[:, 0] - index) <= half_width_slots + 1e-6))
                is_grey = np.allclose(collection.get_facecolor()[0], grey)
                self.assertEqual(is_grey, index not in (0, 2), msg=f"pulse {pulse + 1}")
            waveforms = self.waveform_lines(axis)
            self.assertEqual(len(waveforms), len(shown))
            # Paper-style alignment: the -1..8 ms window is centered on the slot.
            centered = (WAVEFORM_TIME_MS - 3.5) / INTERVAL_MS
            for index, (pulse, line) in enumerate(zip(shown, waveforms)):
                np.testing.assert_allclose(line.get_xdata(), index + centered)
                np.testing.assert_allclose(line.get_ydata(),
                                           np.abs(entry["mean_waveform_mv"][str(pulse)]))
            # Fixed row range, ticked at 0 and the top only.
            top = cartoon.SYNAPTIC_YLIM_TOP[group]
            np.testing.assert_allclose(axis.get_ylim(), (-cartoon.SYNAPTIC_YLIM_PAD * top, top))
            np.testing.assert_allclose(axis.get_yticks(), [0.0, top])
            # Resting-state aCV on pulse 1; induced-state aCV (pulses 5-8 pooled)
            # on the last shown induction pulse (pulse 6 here); noise pooled over all.
            pooled_noise = np.concatenate([entry["trial_noise_mv"][str(p)] for p in range(12)])
            for slot_index, source in ((0, (0,)), (2, (4, 5, 6, 7))):
                pooled = np.concatenate([entry["trial_amplitude_mv"][str(p)] for p in source])
                expected = cmap(norm(cartoon._log_adjusted_cv(pooled, pooled_noise)))
                np.testing.assert_allclose(axis.collections[slot_index].get_facecolor()[0], expected)
        last = figure.axes[-1]
        self.assertEqual([label.get_text() for label in last.get_xticklabels()],
                         ["0", "60", "100", "410", "450"])
        self.assertEqual(last.get_xlabel(), "Time after first presynaptic spike (ms)")
        # Every neighboring pair skips pulses, so each of the four gaps gets a
        # two-stroke break glyph on the bottom axis only.
        self.assertEqual(len(self.break_strokes(last)), 8)
        self.assertEqual(len(self.break_strokes(figure.axes[0])), 0)

    def test_all_twelve_pulses_break_the_axis_only_at_the_recovery_interval(self):
        path = self.root / "amps.json"
        write_full_format(path)
        save_fig = self.render(path, pulses=range(1, 13))
        extra = save_fig.call_args_list[0].kwargs["extra"]
        self.assertIn("8 (140 ms), 9 (390 ms)", extra)
        self.assertIn("pulse 1 and pulse 8", extra)
        figure = save_fig.call_args_list[0].args[0]
        axis = figure.axes[0]
        self.assertEqual(len(axis.collections), 12)
        grey = paper_plot.mpl.colors.to_rgba(cartoon._SYNAPTIC_DOT_COLOR)
        colored = [p for p, c in enumerate(axis.collections)
                   if not np.allclose(c.get_facecolor()[0], grey)]
        self.assertEqual(colored, [0, 7])
        last = figure.axes[-1]
        self.assertEqual([label.get_text() for label in last.get_xticklabels()],
                         [f"{p * INTERVAL_MS:g}" for p in range(8)]
                         + [f"{7 * INTERVAL_MS + 250.0 + k * INTERVAL_MS:g}" for k in range(4)])
        self.assertEqual(len(self.break_strokes(last)), 2)
        self.assertEqual(cartoon.synaptic_pulse_time_ms(8, 50.0, 250.0), 390.0)
        for bad in ((0,), (13,), ()):
            with self.assertRaises(ValueError):
                self.render(path, pulses=bad)

    def test_without_noise_amplitudes_the_colors_use_the_unadjusted_cv(self):
        path = self.root / "amps.json"
        write_full_format(path, with_noise=False, with_classes=False)
        save_fig = self.render(path)
        extra = save_fig.call_args_list[0].kwargs["extra"]
        self.assertIn("UNADJUSTED", extra)
        self.assertIn("depressing: ?", extra)
        loaded = cartoon.load_synaptic_trial_amplitudes(path)
        self.assertFalse(loaded["adjusted"])
        self.assertIsNone(loaded["connections"][0]["noise"])
        self.assertIsNone(loaded["connections"][0]["classes"])

    def test_legacy_metadata_falls_back_to_the_stored_pulses(self):
        path = self.root / "legacy.json"
        write_legacy_format(path)
        save_fig = self.render(path, pulses=(1, 4, 8, 9, 12))
        extra = save_fig.call_args_list[0].kwargs["extra"]
        self.assertIn("legacy", extra)
        self.assertIn("UNADJUSTED", extra)
        figure = save_fig.call_args_list[0].args[0]
        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(len(figure.axes[0].collections), 5)
        dashes = self.waveform_lines(figure.axes[0])
        self.assertEqual(len(dashes), 5)
        np.testing.assert_allclose(dashes[1].get_ydata(), [np.mean([1.0, 0.7, 0.6])] * 2)
        self.assertEqual([label.get_text() for label in figure.axes[0].get_xticklabels()],
                         ["0", "60", "140", "390", "450"])
        loaded = cartoon.load_synaptic_trial_amplitudes(path)
        np.testing.assert_allclose(loaded["connections"][0]["amplitudes"][3], [1.0, 0.7, 0.6])
        self.assertIsNone(loaded["connections"][0]["waveforms"])

    def test_loader_rejects_missing_files_and_malformed_arrays(self):
        with self.assertRaises(FileNotFoundError):
            cartoon.load_synaptic_trial_amplitudes(self.root / "absent.json")
        path = self.root / "bad.json"
        data = write_full_format(path)
        data["connections"][0]["trial_amplitude_mv"]["3"] = []
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            cartoon.load_synaptic_trial_amplitudes(path)
        data = write_full_format(path)
        data["connections"][1]["mean_waveform_mv"]["0"] = [0.1, 0.2]
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            cartoon.load_synaptic_trial_amplitudes(path)
        self.assertEqual(cartoon._parse_pulse_label("Pulse 9 (recovery)"), 8)

    def test_variability_metric_matches_the_paper_definition(self):
        # log( sqrt(var(amp) - var(noise)) / |median| ), population variances.
        self.assertAlmostEqual(cartoon._log_adjusted_cv([1.0, 3.0]), np.log(1.0 / 2.0))
        self.assertAlmostEqual(cartoon._log_adjusted_cv([1.0, 3.0], noise=[-0.5, 0.5]),
                               np.log(np.sqrt(1.0 - 0.25) / 2.0))
        self.assertEqual(cartoon._log_adjusted_cv([1.0, 3.0], noise=[-2.0, 2.0]), -np.inf)
        self.assertTrue(np.isnan(cartoon._log_adjusted_cv([0.3])))
        self.assertTrue(np.isnan(cartoon._log_adjusted_cv([-0.2, 0.2])))

    def test_beeswarm_separates_coincident_points_and_respects_the_width(self):
        offsets = cartoon._beeswarm_offsets([0.0, 0.0, 0.0], marker_px=4.0, max_half_width_px=10.0)
        self.assertEqual(sorted(np.round(offsets, 6)), [-4.0, 0.0, 4.0])
        np.testing.assert_allclose(
            cartoon._beeswarm_offsets([0.0, 10.0, 20.0], marker_px=4.0, max_half_width_px=10.0),
            [0.0, 0.0, 0.0])
        crowded = cartoon._beeswarm_offsets(np.zeros(30), marker_px=4.0, max_half_width_px=6.0)
        self.assertLessEqual(np.abs(crowded).max(), 6.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
