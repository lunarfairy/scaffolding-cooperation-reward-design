# Scaffolding cooperation: reward design

[![Release](https://img.shields.io/badge/release-v1.0.1-blue)](https://github.com/lunarfairy/scaffolding-cooperation-reward-design/releases/tag/v1.0.1)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23277076.svg)](https://doi.org/10.5281/zenodo.23277076)

Code and data for **Reward choice shapes how AI social planners treat free-riders in simulated cooperation networks**.

The repository provides the cooperative-network simulator, reward functions, ES schedule optimization, graph-network PPO, evaluation and analysis. It includes 716 training configurations (592 ES and 124 PPO), their training records, 124 graph-network weights and 5,266 evaluation records.

## Contents

| Location | Contents |
|---|---|
| `code/envtorch.py`, `code/rdp.py` | Simulator, rewards, policies and optimizers |
| `code/run_one.py`, `code/grid.py`, `code/configs/` | Training and experiment configurations |
| `code/eval_all.py`, `code/paired.py` | Common-scenario evaluation and paired statistics |
| `code/marginal.py`, `code/coefprop.py`, `code/propensity.py` | Tie values, coefficient uncertainty and reference scales |
| `code/sim_numpy.py`, `code/calib.py`, `code/refcheck.py` | Independent simulator and calibration checks |
| `code/manuscript_stats.py` | Manuscript summary statistics |
| `inputs/deltas.json` | Reference shifts and scales used by training configurations |
| [data/](data/README.md) | Source tables, evaluations, trained planners and human group outcomes |

Figure-layout scripts are supplied in the accompanying Supplementary Software.

## Setup

Original experiment environment: Python 3.11, PyTorch 2.0.1, CUDA 11.8 and NumPy 1.26 on NVIDIA H20 GPUs.

    python -m venv .venv

Activate with `.\.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Linux/macOS, then install:

    python -m pip install -r requirements.txt

For GPU training, install a PyTorch build compatible with your CUDA environment. On MKL-based Windows installations, `MKL_THREADING_LAYER=SEQUENTIAL` resolves the NumPy/PyTorch OpenMP conflict encountered in local checks.

All commands below run from `code/`.

## Analyse the deposited data

    cd code
    python manuscript_stats.py ../data ../data/human_group_data ../outputs/statistics

Outputs: `numbers_rev.json` (planner summaries, transfer, regret and compatibility), `calibration_mae.csv` (human-comparison errors) and `best_found_k0.5.csv` (optimization comparisons).

To evaluate the deposited trained planners (GPU recommended):

    python eval_all.py launch 1 0 ../outputs/evals ../data/runs_es ../data/runs_gnn
    python paired.py ../outputs/evals ../outputs/paired.csv

## Small training example

This reduced configuration checks execution. Run from `code/` using PowerShell 7:

    $cfg = @{name="demo_es"; policy="table"; obj="coop_mean"; kappa=1.0; seed=0; iters=2; npair=2; games=2; eval_games=16} | ConvertTo-Json -Compress
    python run_one.py $cfg ../outputs/demo

Or Linux/macOS:

    python run_one.py '{"name":"demo_es","policy":"table","obj":"coop_mean","kappa":1.0,"seed":0,"iters":2,"npair":2,"games":2,"eval_games":16}' ../outputs/demo

Outputs are per-run JSON records and PyTorch state dictionaries.

## Re-run the experiments

Full training requires substantial GPU computation. The four configuration files specify the model, reward, optimizer and seed settings:

    python grid.py configs/configs_es.json 1 0 ../inputs/deltas.json ../outputs/runs_es
    python grid.py configs/configs_es2.json 1 0 ../inputs/deltas.json ../outputs/runs_es
    python grid.py configs/configs_gnn.json 1 0 ../inputs/deltas.json ../outputs/runs_gnn
    python grid.py configs/configs_gnn2.json 1 0 ../inputs/deltas.json ../outputs/runs_gnn
    python eval_all.py launch 1 0 ../outputs/evals ../outputs/runs_es ../outputs/runs_gnn
    python paired.py ../outputs/evals ../outputs/paired.csv

The `1 0` arguments set one worker on GPU 0. Evaluation uses reporting set A (seed 12345) and independent selection set B (24680): 20,000 games for own-condition evaluations and 10,000 for frozen-policy transfer. Paired comparisons use the same evaluation batch size (4,000) across planners. `eval_all.py` generates the per-game arrays needed by `paired.py`.

`RDP_N_OWN` and `RDP_N_CROSS` can reduce evaluation sizes for execution checks; use their defaults for manuscript experiments. Training results can vary with hardware and software.

Additional analyses:

    python propensity.py ../outputs/deltas_regenerated.json
    python marginal.py ../outputs/runs_es ../outputs/marg 20000
    python coefprop.py ../outputs/coef 20000 1000
    python calib.py ../inputs/deltas.json ../outputs/calib.json 8000
    python refcheck.py ../outputs/refcheck.json 8000
    python manuscript_stats.py ../outputs ../data/human_group_data ../outputs/statistics

`code/make_configs.py` regenerates revision configuration lists in its working directory. For thresholds, uncertainty fields and data provenance, see the [data README](data/README.md).

## Citation and licence

Lu, J. & Tu, C. (2026). *Scaffolding cooperation: reward design - core experiment and analysis code* (v1.0.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.23277076

The fixed v1.0.0 and v1.0.1 archives contain code only. Supporting data were added on 29 October 2026; identify the data version by its Git commit. Machine-readable software citation metadata is in [CITATION.cff](CITATION.cff).

Code: [MIT](LICENSE). Human group data: [McKee et al. (2023)](https://doi.org/10.1038/s41562-023-01686-7), available from [OSF](https://osf.io/8ahkg/) under CC BY 4.0; see [attribution and licence](data/human_group_data/README.md). Questions: GitHub issues.
