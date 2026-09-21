"""PCA export and rendering checks using synthetic fixed points only."""

import ast
import copy
import inspect
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.decomposition import PCA

import _bootstrap  # noqa: F401
from core import fixed_point_pca as export
import paper_plot as paper


class FixedPointPCATests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.data = {"aname": "synthetic", "rule": "delayanti", "results": {
            period: {"fixed_M": rng.normal(size=(8, 3, 4)),
                     "fixed_WM": rng.normal(size=(8, 3, 4)),
                     "fixed_hidden": rng.normal(size=(8, 5)),
                     "is_fixed": np.arange(8) % 2 == 0}
            for period in ("longdelay", "longstimulus")}}
        self.data["angles"] = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        for period, entry in self.data["results"].items():
            entry.update(stim=np.arange(8), period_title=period,
                         rel_step=np.full(8, 0.01), rel_tol=0.05)

    def test_bases_match_original_fits_and_local_projections(self):
        artifact = export.fit_fixed_point_bases(self.data)
        for period, entries in artifact["bases"].items():
            for representation, record in entries.items():
                with self.subTest(period=period, representation=representation):
                    raw = self.data["results"][period][representation]
                    flat = raw.reshape(len(raw), -1)
                    reference = PCA(n_components=2, random_state=0).fit(flat)
                    np.testing.assert_allclose(record["mean"], reference.mean_)
                    np.testing.assert_allclose(record["components"], reference.components_)
                    local = PCA(n_components=2, random_state=0).fit_transform(flat)
                    np.testing.assert_allclose(record["fit_projection"], local)
                    self.assertEqual(record["source_rule"], "delayanti")
                    self.assertEqual(record["n_samples"], 8)

    def test_sidecar_preserves_source_and_skips_unchanged_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixed_points_grad_synthetic.pkl"
            original = pickle.dumps(self.data)
            source.write_bytes(original)
            stat = source.stat()
            destination = export.export_fixed_point_pca(source)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(source.stat().st_mtime_ns, stat.st_mtime_ns)
            with np.load(destination, allow_pickle=True) as saved:
                artifact = saved["artifact"].item()
            self.assertEqual(artifact["source"], export.source_signature(source))
            with patch.object(export, "fit_fixed_point_bases", side_effect=AssertionError("refit")):
                self.assertEqual(export.export_fixed_point_pca(source), destination)

    def test_paper_uses_saved_basis_without_refitting(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixed_points_grad_synthetic.pkl"
            source.write_bytes(pickle.dumps(self.data))
            export.export_fixed_point_pca(source)
            data = paper._load_pkl_or_skip(source)
            raw = self.data["results"]["longdelay"]["fixed_hidden"]
            reference = PCA(n_components=2, random_state=0).fit(raw).transform(raw)
            with patch.object(PCA, "fit", side_effect=AssertionError("paper refit")), \
                    patch.object(PCA, "fit_transform", side_effect=AssertionError("paper refit")):
                basis = paper._load_period_grad_fp_basis(data, "fixed_hidden")
                np.testing.assert_allclose(basis.transform(raw), reference)
                self.assertIsNone(paper._load_period_grad_fp_basis(data, "fixed_hidden", "missing_period"))

    def test_missing_and_stale_sidecars_do_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixed_points_grad_synthetic.pkl"
            source.write_bytes(pickle.dumps(self.data))
            data = paper._load_pkl_or_skip(source)
            self.assertIsNone(paper._load_period_grad_fp_basis(data, "fixed_hidden"))
            export.export_fixed_point_pca(source)
            self.assertIsNotNone(paper._load_period_grad_fp_basis(data, "fixed_hidden"))
            source.write_bytes(source.read_bytes() + b"changed")
            self.assertIsNone(paper._load_period_grad_fp_basis(data, "fixed_hidden"))

    def test_two_task_reference_basis_is_not_replaced_by_first_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for rule in ("delaygo", "delayanti"):
                source = Path(directory) / f"fixed_points_grad_synthetic_{rule}.pkl"
                data = copy.deepcopy(self.data)
                data["rule"] = rule
                if rule == "delaygo":
                    for entry in data["results"].values():
                        for representation in export.REPRESENTATIONS:
                            entry[representation] += 10
                source.write_bytes(pickle.dumps(data))
                export.export_fixed_point_pca(source)
                paths.append((rule, source))
            shared = paper._twotask_shared_fp_bases(paths, "test", period="longstimulus")
            self.assertEqual(set(shared), set(export.REPRESENTATIONS))
            for representation, basis in shared.items():
                raw = self.data["results"]["longstimulus"][representation]
                np.testing.assert_allclose(basis.mean_, raw.reshape(len(raw), -1).mean(axis=0))
            self.assertEqual(paper._twotask_shared_fp_bases(paths[:1], "missing reference"), {})

    def test_randomized_solver_matches_old_basis(self):
        raw = np.random.default_rng(3).normal(size=(30, 600))
        data = {"results": {"longdelay": {"fixed_hidden": raw}}}
        record = export.fit_fixed_point_bases(data)["bases"]["longdelay"]["fixed_hidden"]
        reference = PCA(n_components=2, random_state=0).fit(raw)
        np.testing.assert_allclose(record["components"], reference.components_)

    def test_single_task_rnn_and_dense_angle_renderers_do_not_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixed_points_grad_synthetic.pkl"
            source.write_bytes(pickle.dumps(self.data))
            sidecar = export.export_fixed_point_pca(source)
            original = sidecar.read_bytes()
            data = paper._load_pkl_or_skip(source)
            with patch.object(PCA, "fit", side_effect=AssertionError("paper refit")), \
                    patch.object(PCA, "fit_transform", side_effect=AssertionError("paper refit")), \
                    patch.object(paper, "_ensure_out_dir"), \
                    patch.object(paper, "_save_fig") as save:
                try:
                    for representation in export.REPRESENTATIONS:
                        paper._render_grad_fixed_points(data, representation, Path(directory) / "2d.png")
                        paper._render_grad_fixed_points_3d(data, representation, Path(directory) / "3d.png")
                    paper._render_interp_fixed_points(data, Path(directory) / "angles.png")
                    self.assertEqual(save.call_count, 7)
                    for call in save.call_args_list:
                        figure = call.args[0]
                        self.assertTrue(figure.axes)
                        figure.canvas.draw()
                    self.assertEqual(sidecar.read_bytes(), original)
                finally:
                    paper.plt.close("all")

    def test_two_task_and_alpha_drivers_use_saved_reference_bases(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for rule in ("delaygo", "delayanti"):
                source = Path(directory) / f"fixed_points_grad_synthetic_{rule}.pkl"
                source.write_bytes(pickle.dumps(dict(self.data, rule=rule)))
                export.export_fixed_point_pca(source)
                paths.append((rule, source))
            with patch.object(paper, "_twotask_grad_fp_paths", return_value=paths), \
                    patch.object(paper, "_load_twotask_glob_or_skip", return_value={}), \
                    patch.object(PCA, "fit", side_effect=AssertionError("paper refit")), \
                    patch.object(PCA, "fit_transform", side_effect=AssertionError("paper refit")):
                renderer = Mock()
                paper._plot_two_task_grad_fp_combined("test", "test", renderer, with_pc_label=True)
                self.assertEqual(renderer.call_count, 3)
                for call in renderer.call_args_list:
                    representation, basis = call.args[1], call.args[3]
                    self.assertEqual(call.kwargs["pc_label"], "Stimulus")
                    self.assertIn("_stimpc_", call.args[2].name)
                    raw = self.data["results"]["longstimulus"][representation]
                    np.testing.assert_allclose(basis.mean_, raw.reshape(len(raw), -1).mean(axis=0))
                renderer.reset_mock()
                paper._plot_two_task_interp_alpha_fp(renderer, "alpha", "test")
                self.assertEqual(renderer.call_count, 3)
                for call in renderer.call_args_list:
                    raw = self.data["results"]["longdelay"][call.args[1]]
                    np.testing.assert_allclose(call.args[3].mean_, raw.reshape(len(raw), -1).mean(axis=0))

    def test_two_task_3d_can_show_all_candidates_solid(self):
        entry = {
            "stim": np.array([0, 1]),
            "is_fixed_strict": np.array([True, False]),
            "period_title": "Stimulus",
            "is_diagonal": True,
        }
        results = {"longstimulus": entry}
        projection = {"longstimulus": np.array([[0.0, 0.0], [1.0, 1.0]])}
        z_values = {"longstimulus": np.zeros(2)}
        fig = paper.plt.figure()
        try:
            with patch.object(paper, "_scatter_grad_fp") as scatter:
                paper._draw_grad_fp_3d_row(
                    fig, results, ["longstimulus"], projection, z_values, {},
                    n_stim=2, lim=1.1, zmax=1.0, n_rows=1, row_idx=0,
                    n_col=1, solid_candidates=True)
            np.testing.assert_array_equal(scatter.call_args.args[4], [True, True])
        finally:
            paper.plt.close(fig)

    def test_interp_plot_distinguishes_strict_approximate_and_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixed_points_grad_synthetic.pkl"
            data = copy.deepcopy(self.data)
            entry = data["results"]["longdelay"]
            entry["rel_step_undamped"] = np.array(
                [0.005, 0.010, 0.011, 0.030, 0.050, 0.051, 0.2, np.nan])
            entry["rel_tol_undamped"] = 0.01
            entry["approx_rel_tol_undamped"] = 0.05
            source.write_bytes(pickle.dumps(data))
            export.export_fixed_point_pca(source)
            loaded = paper._load_pkl_or_skip(source)
            with patch.object(paper, "_ensure_out_dir"), \
                    patch.object(paper, "_save_fig") as save:
                paper._render_interp_fixed_points(
                    loaded, Path(directory) / "interp.png", n_trained=8)

        figure = save.call_args.args[0]
        try:
            self.assertIn("2 strict / 3 approximate / 3 failed",
                          figure.axes[0].get_title())
            threshold_lines = [
                line.get_label() for line in figure.axes[1].lines
                if "threshold" in line.get_label()
            ]
            self.assertEqual(len(threshold_lines), 2)
        finally:
            paper.plt.close(figure)

    def test_paper_contains_no_estimator_refits(self):
        tree = ast.parse(inspect.getsource(paper))
        refits = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr in ("fit", "fit_transform")]
        self.assertEqual(refits, [])

    def test_backfill_directory_discovery_does_not_treat_sidecars_as_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "run"
            nested.mkdir()
            sources = [root / "fixed_points_grad_single.pkl",
                       nested / "fixed_points_grad_two_delayanti.pkl",
                       nested / "fixed_points_hidden_rnn.pkl"]
            for path in sources:
                path.write_bytes(pickle.dumps(self.data))
            (root / "interp_fixed_points_two.pkl").write_bytes(pickle.dumps({"alphas": [0, 1]}))
            export.main([str(root)])
            self.assertEqual(len(list(root.rglob("*.pca.npz"))), 3)
            with patch.object(export, "fit_fixed_point_bases", side_effect=AssertionError("unnecessary refit")):
                export.main([str(root)])


if __name__ == "__main__":
    unittest.main()
