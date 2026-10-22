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
        # cartoon imported `_save_fig` by name, so its own binding is the one to
        # patch: the figure is saved through cartoon._save.
        save_fig = Mock()
        with patch.object(cartoon, "_save_fig", save_fig):
            cartoon.plot_synaptic_train_amplitudes(out_dir=self.root, data_path=data_path,
                                                   **kwargs)
        for call in save_fig.call_args_list:
            self.addCleanup(paper_plot.plt.close, call.args[0])
        return save_fig

    @staticmethod
    def waveform_lines(axis):
        return [line for line in axis.lines if line.get_color() == "black" and line.get_zorder() == 100]

    def test_default_draws_selected_pulses_in_slots_labeled_with_real_time(self):
        path = self.root / "amps.json"
        data = write_full_format(path)
        # One depressing-row dot above that row's tick top: the range must
        # follow it instead of clipping it.
        depressing = next(entry for entry in data["connections"] if entry["group"] == "depressing")
        depressing["trial_amplitude_mv"]["0"][0] = 1.3
        path.write_text(json.dumps(data))
        save_fig = self.render(path)
        names = [call.args[1].name for call in save_fig.call_args_list]
        self.assertEqual(names, ["cartoon_synaptic_train_amplitudes.png"])
        extra = save_fig.call_args_list[0].kwargs["extra"]
        self.assertIn("MEASURED", extra)
        self.assertIn("pulses 1 (0 ms), 4 (60 ms), 6 (100 ms), 10 (410 ms), 12 (450 ms)", extra)
        self.assertNotIn("log CV", extra)
        self.assertIn("depressing: L5 ET\u2192Pv", extra)
        figure = save_fig.call_args_list[0].args[0]
        self.assertEqual(len(figure.axes), 3)
        shown = [0, 3, 5, 9, 11]                      # 0-based pulses of SYNAPTIC_PULSES
        by_group = {entry["group"]: entry for entry in data["connections"]}
        self.assertNotIn("clipped", extra)
        grey =paper_plot.mpl.colors.to_rgba(cartoon._SYNAPTIC_DOT_COLOR)
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
                # Every dot in the same grey: no variability coloring.
                np.testing.assert_allclose(collection.get_facecolor()[0], grey)
            waveforms = self.waveform_lines(axis)
            self.assertEqual(len(waveforms), len(shown))
            # Paper-style alignment: the -1..8 ms window is centered on the slot.
            centered = (WAVEFORM_TIME_MS - 3.5) / INTERVAL_MS
            for index, (pulse, line) in enumerate(zip(shown, waveforms)):
                np.testing.assert_allclose(line.get_xdata(), index + centered)
                np.testing.assert_allclose(line.get_ydata(),
                                           np.abs(entry["mean_waveform_mv"][str(pulse)]))
            # Ticked at 0 and the row's fixed top; the range pads below zero and
            # reaches past the top when a dot does, so none is cut off.
            top = cartoon.SYNAPTIC_YLIM_TOP[group]
            pad = cartoon.SYNAPTIC_YLIM_PAD * top
            dots = max(max(entry["trial_amplitude_mv"][str(p)]) for p in shown)
            waves = max(float(np.abs(entry["mean_waveform_mv"][str(p)]).max()) for p in shown)
            np.testing.assert_allclose(axis.get_ylim(), (-pad, max(top, dots, waves) + pad))
            np.testing.assert_allclose(axis.get_yticks(), [0.0, top])
            if group == "depressing":
                self.assertGreater(dots, top)      # the overshooting dot exercised the extension
        last = figure.axes[-1]
        self.assertEqual([label.get_text() for label in last.get_xticklabels()],
                         ["0", "60", "100", "410", "450"])
        self.assertEqual(last.get_xlabel(), "Time after first presynaptic spike (ms)")
        # No break glyph: the uneven gaps are carried by the tick labels alone,
        # so the bottom axes holds nothing but its waveforms.
        self.assertEqual(len(last.lines), len(self.waveform_lines(last)))
        # The key is saved beside the figure as its own file.
        self.assertTrue((self.root / "cartoon_synaptic_train_amplitudes_legend.png").is_file())

    @staticmethod
    def legend_entries(line_kinds):
        figure = cartoon._synaptic_legend_figure(line_kinds)
        legend = figure.legends[0]
        entries = [(text.get_text(), line) for text, line in zip(legend.get_texts(), legend.get_lines())]
        paper_plot.plt.close(figure)
        return entries

    def test_legend_names_the_dot_and_the_trial_mean_mark(self):
        grey = paper_plot.mpl.colors.to_rgba(cartoon._SYNAPTIC_DOT_COLOR)
        (dot_label, dot), (line_label, line) = self.legend_entries(("waveform",))
        self.assertEqual(dot_label, "Single-trial PSP amplitude")
        self.assertEqual(dot.get_marker(), "o")
        self.assertEqual(dot.get_linestyle(), "None")
        np.testing.assert_allclose(paper_plot.mpl.colors.to_rgba(dot.get_markerfacecolor()), grey)
        self.assertEqual(line_label, "Trial-mean PSP waveform")
        self.assertEqual(line.get_color(), "black")
        # The legacy file draws a mean-amplitude dash, and the key says so.
        self.assertEqual([label for label, _ in self.legend_entries(("dash",))],
                         ["Single-trial PSP amplitude", "Trial-mean PSP amplitude"])

    def test_legacy_render_saves_the_dash_key(self):
        path = self.root / "legacy.json"
        write_legacy_format(path)
        with patch.object(cartoon, "_save_synaptic_legend") as save_legend:
            self.render(path, pulses=(1, 4, 8, 9, 12))
        save_legend.assert_called_once_with(self.root, ["dash"])

    def test_all_twelve_pulses_label_the_recovery_interval_in_real_time(self):
        path = self.root / "amps.json"
        write_full_format(path)
        save_fig = self.render(path, pulses=range(1, 13))
        extra = save_fig.call_args_list[0].kwargs["extra"]
        self.assertIn("8 (140 ms), 9 (390 ms)", extra)
        figure = save_fig.call_args_list[0].args[0]
        axis = figure.axes[0]
        self.assertEqual(len(axis.collections), 12)
        last = figure.axes[-1]
        self.assertEqual([label.get_text() for label in last.get_xticklabels()],
                         [f"{p * INTERVAL_MS:g}" for p in range(8)]
                         + [f"{7 * INTERVAL_MS + 250.0 + k * INTERVAL_MS:g}" for k in range(4)])
        self.assertEqual(len(last.lines), len(self.waveform_lines(last)))
        self.assertEqual(cartoon.synaptic_pulse_time_ms(8, 50.0, 250.0), 390.0)
        for bad in ((0,), (13,), ()):
            with self.assertRaises(ValueError):
                self.render(path, pulses=bad)

    def test_files_without_noise_or_classes_still_load_and_draw(self):
        path = self.root / "amps.json"
        write_full_format(path, with_noise=False, with_classes=False)
        save_fig = self.render(path)
        extra = save_fig.call_args_list[0].kwargs["extra"]
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
