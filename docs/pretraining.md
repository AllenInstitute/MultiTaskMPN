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
python paper_plot.py pretraining --pretraining-groups motifs
# pretraining: motifs plus DelayAnti and DelayPro (default)
python paper_plot.py pretraining --pretraining-groups all
# pretraining: the runs trained with lambda = 0.9 (m_time_scale 400 ms)
python paper_plot.py pretraining --pretraining-lambda 0.9
```

`--pretraining-groups {motifs,all}` controls the conditions shown in every
pretraining paper figure: backbone probe, transfer speed, learning trajectory,
rule vectors, principal angles, and both aggregate CVE figures. `motifs` selects
only Relevant/Irrelevant motifs; `all` also includes the DelayAnti/DelayPro
single-task controls and remains the default. The selection applies to both
combined and per-seed caches, including the CVE self-reference, and can be
combined with `--pretraining-bound mod2`, `--no-legend`, or `--only`, for example:
`python paper_plot.py --only learning_trajectory --pretraining-groups motifs`.
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

`python paper_plot.py --only learning_trajectory_linear` exports
`learning_trajectory_linear_n.png`: the same seed and mean curves as
`learning_trajectory_n.png`, with a linear iteration axis instead of a log axis.
Both are included in `pretraining` mode and honor the group/bound options.
`rule_vectors_n.png` is drawn at the `learning_trajectory` size (the six-bar
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
