# Pretraining to post-training transfer

The two-stage transfer experiment (`pretrain/pretraining.py`), its analysis
scripts and the `pretraining` figures of `paper_plot.py`.

```bash
# pretraining transfer: each run adds N_TRIALS (5) new seeds of PRETRAIN_RULESET
# (fdgo_delaygo) at the configured lambda (0.9); seeds with a completed run are
# never redrawn, so rerun it to grow the cohort, then refresh its analysis caches
python pretrain/pretraining.py
python pretrain/pretraining_analysis.py --feature L21e3lam0.9
python pretrain/pretraining_post.py --feature L21e3lam0.9

# pretraining: Relevant and Irrelevant motifs only
python figures/paper_plot.py pretraining --pretraining-groups motifs
# pretraining: motifs plus DelayAnti and DelayPro (default)
python figures/paper_plot.py pretraining --pretraining-groups all
# pretraining: the runs trained with lambda = 0.9 (m_time_scale 400 ms)
python figures/paper_plot.py pretraining --pretraining-lambda 0.9
```

`--pretraining-groups {motifs,all}` controls the conditions shown in every
pretraining paper figure: backbone probe, transfer speed, learning trajectory,
rule vectors, principal angles, and both aggregate CVE figures. `motifs` selects
only Relevant/Irrelevant motifs; `all` also includes the DelayAnti/DelayPro
single-task controls and remains the default. The selection applies to both
combined and per-seed caches, including the CVE self-reference, and can be
combined with `--pretraining-bound mod2`, `--no-legend`, or `--only`, for example:
`python figures/paper_plot.py --only learning_trajectory --pretraining-groups motifs`.
Other experiment modes are unaffected. Output filenames are unchanged, so
running the other group selection replaces the same figure; `--only` preserves
unrelated figures, while mode runs clear the top-level `paper_plot/` outputs (see [docs/multitask_analysis.md](multitask_analysis.md)).

The pretraining figures read one variant of the `pretrain/pretraining.py`
runs (dmpn, hidden 200, L2 1e-3, batch 128), chosen by two options that select
the matching files in `pretraining_analysis/` by name and never mix variants:
`--pretraining-bound {mod1,mod2}` picks the M bound ([-1, 1], the default, or
[-2, 2], tagged `mb2`), and `--pretraining-lambda {0.99,0.9}` picks the M
retention per 40 ms step. `0.99` (default) is the historical cohort trained with
`m_time_scale` 4000 ms, whose file names carry no tag; `0.9` selects the runs
trained with 400 ms, tagged `lam0.9` (for example
`fdanti_delaygo_dmpn_seed3_+hidden200+L21e3lam0.9+batch128+angle_result.pkl`).
The two options combine (`L21e3mb2lam0.9`). Every pretraining figure aggregates
all cached seeds of the selected variant, and the run banner prints the
selected lambda, `m_time_scale` and feature string. Output filenames do not
encode the variant, so rendering another selection replaces the same figures.
Every figure of the mode is saved as `pretraining_<figure>.png` (for example
`--only aggregate_cve_response` writes `pretraining_aggregate_cve_response.png`,
with the `_n` suffix when legends are on), as the other experiment families
carry their `onetask_`, `twotask_` and `multitask_` prefixes; before
2026-11-14 only `pretraining_principal_angles` had the prefix.

`python figures/paper_plot.py --only learning_trajectory_linear` exports
`pretraining_learning_trajectory_linear_n.png`: the seeds and condition means of
`pretraining_learning_trajectory_n.png` on one linear iteration axis to 20000
(`_LEARNING_TRAJECTORY_LINEAR_XMAX`), a quarter of the 80000 training
iterations. The slow conditions are still climbing at its right edge: the
irrelevant motif's mean is 87.3 % at 20000, 91.6 % at 40000 and 92.7 % at the
end of training at λ = 0.99 (83.4, 93.2 and 96.4 % at λ = 0.9), so the gap to
the relevant motif there is wider than the final one; the log figure shows
the whole of training. On that axis the first 1000 iterations
(`_LEARNING_TRAJECTORY_LINEAR_SPLIT`), where the relevant motif saturates and
the irrelevant motif does most of its climbing, are a sliver at the left
edge, so an inset in the lower right draws them at full scale on the same
accuracy axis; a box around that range and two connectors tie it to the
inset. The inset and the rest of the range are resampled on their own linear
grids, so their sample points differ from the log figure's log-spaced grid,
and the saved message names the two ranges; the legend sits above the axes.
Until 2026-11-14 the figure broke the axis into two panels (0 to 1000 and
1000 to 40000) instead; the first inset version also ran to 40000. Both
figures are included in `pretraining` mode and honor the group/bound options.
`pretraining_backbone_probe_n.png` is as wide as its columns need (two
conditions with `motifs`, four with `all`) at the height of the other
pretraining panels, and names the probe in its y label instead of a title;
until 2026-11-14 it was a fixed 3.4 x 2.0 in.
`pretraining_rule_vectors_n.png` is drawn at the `learning_trajectory` size (the six-bar
`all` layout keeps that height and width per bar). Each bar keeps its two-line
"MemoryAnti ↔ partner" tick label, in a 5 pt font so that neighbouring labels
do not collide at that width.
For `--pretraining-groups motifs`, the four bars form two pairs (2 | 2) set
apart by a wider gap and a dark vertical divider; `all` keeps its evenly spaced
layout.

Both `pretrain/pretraining_analysis.py` and `pretrain/pretraining_post.py` write
analysis data to `pretraining_analysis/`, pooled figures to `pretrain/fig/`, and
per-seed figures to `pretrain/fig_seed/`. The latter includes per-checkpoint
PCA/sanity checks, single-checkpoint diagnostics, and seed-filtered accuracy plots.
Existing outputs with matching filenames are overwritten. Training inputs and
checkpoints remain in `pretraining/`; `paper_plot.py` reads the centralized
analysis data and continues to save publication figures in `paper_plot/`.
