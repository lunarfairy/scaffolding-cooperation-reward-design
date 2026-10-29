# Source data and trained planners

Data supporting the five main figures, four Extended Data figures and four Extended Data tables. [figure_data_map.csv](figure_data_map.csv) maps display items to inputs and processing functions. Figure-layout functions are in the accompanying Supplementary Software.

## Files

| Files | Contents |
|---|---|
| `evals/` | 21 JSONL files with 5,266 evaluation records, including trajectories and recommendation arrays |
| `evaluation_scalars.csv` | Scalar fields exported from the evaluation records |
| `paired.csv` | Objective values, held-out best-family selection, shortfalls, Monte Carlo errors, bootstrap intervals and normalized regret |
| `marg_summary.csv`, `marg_ties.csv.gz` | Tie-value summaries and tie-level records |
| `coef_draws.csv`, `coef_strata.csv` | Coefficient draws, one-step value curves and stratified diagnostics |
| `calib.json`, `refcheck.json`, `deltas.json` | Calibration, simulator comparisons and reference shifts/scales |
| `numbers_rev.json` | Manuscript summary statistics |
| `run_manifest.csv`, `training_logs.jsonl` | Training configurations and logs |
| `runs_es/` | 592 ES run files, including learned schedule parameters |
| `runs_gnn/`, `weights_gnn/` | 124 PPO run files and matching PyTorch weights, paired by filename stem |
| `human_group_data/` | Three released human group-outcome CSVs, with source and licence information |

Numerical outputs and weights retain their original values; `evaluation_scalars.csv` and `training_logs.jsonl` consolidate existing records. [CHECKSUMS.sha256](CHECKSUMS.sha256) lists SHA-256 digests.

## Fields and uncertainty

- `name`: planner; `kind`: policy class; `family`: objective or rule family.
- `seed_train`: training seed. `seed`: scenario set, with 12345 for reporting (A) and 24680 for independent selection (B).
- `train_cond`, `eval_cond`: training and evaluation conditions; `n`, `n_games`: sample sizes.
- `_se` fields: Monte Carlo standard errors. Across-training-seed variability is reported separately in the manuscript.
- `rec_rates`, `acc_rates`, `chg_rates`, `rec_opp`: pair-type indices 0 = DD, 1 = CD, 2 = CC; tie-status indices 0 = absent/add, 1 = present/delete. Pooled rates use opportunity weighting, as implemented in `../code/manuscript_stats.py`.

Tie-value break-even thresholds use a 0.005 grid over 0–1.5 and interpolate the first negative-to-non-negative crossing. The threshold is 0 when the curve is already non-negative at zero, and undefined when there is no crossing. Intervals use 2,000 bootstrap resamples: seed 17 for break-even thresholds and seed 7 for effects. `break_even_boot_nan` gives the fraction of undefined bootstrap thresholds; percentile limits use defined samples.

Coefficient uncertainty uses the same grid with 1,000 coefficient draws, distinct from bootstrap resampling. `value_k1` and `value_k05` give each draw's values at kappa = 1 and 0.5. Same-family shortfall confidence intervals are zero.

## Use

From `code/`, after installing the repository requirements:

    python manuscript_stats.py ../data ../data/human_group_data ../outputs/statistics

For new evaluations and paired bootstrap calculations, use `eval_all.py` and `paired.py` as shown in the [main README](../README.md). Per-game arrays are generated during evaluation and are not included here.

## Source and version

Human group outcomes originate from [McKee et al. (2023)](https://osf.io/8ahkg/) under CC BY 4.0; [source details and original licence](human_group_data/README.md) are retained. Scientific code uses the [MIT licence](../LICENSE).

The data were added on 29 October 2026 and are versioned by Git commit. The earlier v1.0.0/v1.0.1 Zenodo archives contain code only.
