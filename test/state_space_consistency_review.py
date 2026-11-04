"""Manual, synthetic checks of state-space producer/reader consistency.

Writes only temporary files; does not modify scientific caches or figures.
"""
import contextlib
import io
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import paper_plot
import state_space_shift as shift
import state_space_structure as structure
import state_space_vs_lesion as lesion
from test_state_space_structure import _pca_cache


def result_entry(schema, r):
    return {"schema_version": schema, "l2_info": 1e-4,
            "m_alignment": "post_update" if schema >= 3 else "pre_update",
            "rval_dict": {"hidden": (r, 1.0, 0.001)},
            "scatter": {"hidden": {"permutation": {"p_value": 0.01}}}}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        pca = _pca_cache("run", 1e-4, 0)
        result = structure.analyze_run(pca, n_permutations=10)
        summary = structure.summarize_run(result)
        path = structure.structure_cache_path("run", directory)
        with path.open("wb") as stream:
            pickle.dump(result, stream)
        with patch.object(paper_plot, "STATE_SPACE_DIR", directory):
            plotted = paper_plot._load_state_space_structure("run")
        loaded = lesion.load_structure_distances(path)
        print(json.dumps({"check": "alignment_propagation",
                          "source_alignment": pca.get("m_alignment"),
                          "structure_alignment": result.get("m_alignment"),
                          "summary_alignment": summary.get("m_alignment"),
                          "aligned_structure_accepted_by_plot": plotted is not None,
                          "aligned_structure_accepted_by_lesion": loaded is not None}))

        results = {"old": result_entry(2, 0.9), "current": result_entry(3, 0.2)}
        with (directory / shift.RESULTS_NAME).open("wb") as stream:
            pickle.dump(results, stream)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            shift.summarize(directory)
            grouped = paper_plot._state_space_r_values_by_l2(results, ["hidden"])
        means = [line.strip() for line in captured.getvalue().splitlines()
                 if "R-values: mean=" in line]
        print(json.dumps({"check": "legacy_filter_consistency",
                          "source_summary": means,
                          "paper_plot_kept": grouped[1e-4]["hidden"]}))

        untrained = result_entry(1, 0.6)
        untrained.update(untrained=True, init_seed=0, config_source="run", l2_info=None,
                         experiment="modulation_transplant", evaluation_seed=42,
                         m_alignment=shift.M_ALIGNMENT)
        path = directory / "untrained_transplant_results.pkl"
        with path.open("wb") as stream:
            pickle.dump({"run_init0": untrained}, stream)
        with patch.object(paper_plot, "_load_best_state_space_trial", return_value=None):
            group = paper_plot._untrained_r_values(
                ["hidden"], path, paper_plot.STATE_SPACE_TRANSPLANT_SCHEMA,
                paper_plot._transplant_untrained_command,
                trained_results={"run": {**untrained, "untrained": False, "l2_info": 1e-4}})
        print(json.dumps({"check": "standalone_transplant_untrained_panel",
                          "valid_untrained_file_exists": path.exists(),
                          "panel_returned_without_original_PCA_ranking": group is not None}))


if __name__ == "__main__":
    main()
