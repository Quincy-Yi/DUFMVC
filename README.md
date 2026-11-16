# DU-FMVC: Dual-Uncertainty-Aware Federated Incomplete Multi-View Clustering

PyTorch implementation of a vertical federated incomplete multi-view clustering framework.  
This project jointly addresses two key challenges in federated incomplete multi-view clustering:

- data-level uncertainty caused by missing views,
- model-level uncertainty caused by heterogeneous clients.

At the client side, the framework uses a Collaborative Adaptive Reconstruction module to initialize and iteratively refine missing-view features. At the server side, it uses quality-aware aggregation and inertia-based target updates to generate more stable global clustering supervision.

## Overview

The training pipeline in this repository contains two stages:

- **Pretraining**: learn stable local graph representations for each view.
- **Federated collaborative training**: aggregate multi-view embeddings on the server, build global target distributions, and feed them back to each client for local optimization.

The main implementation files are:

- `main.py`: end-to-end training entry.
- `generate_incomplete_data.py`: incomplete-view generation, view imputation, and graph construction.
- `model.py` and `layer.py`: graph encoders and layers.
- `pretrain.py` and `train.py`: local pretraining and federated training routines.
- `evaluation.py`: ACC, NMI, ARI, and F1 evaluation.

## Datasets

The repository currently includes four public datasets in `datasets/`:

| Dataset | Samples | Views | Clusters | Default missing rates in `main.py` |
| --- | ---: | ---: | ---: | --- |
| `BDGP_4view` | 685 | 4 | 5 | `[0.1, 0.3, 0.5, 0.7]` |
| `Scene-15` | 4485 | 3 | 15 | `[0.1, 0.5, 0.7]` |
| `Reuters` | 1500 | 5 | 6 | `[0.1, 0.2, 0.2, 0.1, 0.1]` |
| `Caltech101-7` | 1474 | 6 | 7 | `[0.1, 0.1, 0.2, 0.2, 0.1, 0.1]` |

You can also specify custom missing rates with `--missrates`.

## Run the code by

```bash
python main.py
```

Example runs:

```bash
python main.py --name BDGP_4view
python main.py --name Scene-15
python main.py --name Reuters
python main.py --name Caltech101-7
```

Different clients/views can use different missing rates. For example:

```bash
python main.py --name BDGP_4view --missrates 0.1 0.3 0.5 0.7
```

The default experiment mode is `full`. After every communication round, the
program refreshes two files in that run's
`log_new/<dataset>_<mode>_<timestamp>/` folder:

- `<dataset>_<rate1>-<rate2>-....xlsx`: every client's `U_d` and `U_m`, plus
  their global arithmetic means for each round.
- `<dataset>_<rate1>-<rate2>-....png`: normalized global-mean `U_d` and `U_m`
  together with raw server-fusion ACC in one dual-axis chart.

The workbook contains `Uncertainty_History` (one row per round), `Per_Client`
(long-form client records with the corresponding missing rate), and `Metadata`.
The verified PNG curve is also embedded in `Uncertainty_History` for convenience.
`Uncertainty_History` records synchronized PRE/POST Ud and Um values for every
client and their global means, together with recovery-gate statistics, the
uncertainty-weighted model-alignment loss, QAA diagnostics, ACC, NMI, ARI, and
F1. The PNG uses
the POST global means. For visualization only, each uncertainty curve is divided
by its initial five-round reference mean; faint lines show raw rounds and bold lines show a centered
3-round moving average. Raw ACC uses an adaptive right axis bounded by `[0, 1]`.
The workbook keeps every original measurement. Communication rounds are numbered
from 1 in both the workbook and the plot.

Uncertainty ablations use the same training entry and code path:

```bash
python main.py --name BDGP_4view --experiment-mode original
python main.py --name BDGP_4view --experiment-mode full
```

All KMeans calls in pretraining, client-quality estimation, and server
aggregation use `cuml.cluster.KMeans` through `gpu_kmeans.py`. There is no CPU
KMeans fallback. The adapter keeps `init="k-means++"`, preserves the configured
`n_init`, fixes the seed with `--kmeans-seed` (default `1018`), and converts
labels/centers back to NumPy for the existing alignment code.

Available modes are `original`, `no_uncertainty`, `ud_only`, `um_only`, `car_gate`,
`weighted_alignment`, and `full`. The paper-aligned defaults are 20 communication
rounds, 50 local epochs, 100 pretraining epochs, `B=5`, `eta_1=0.4`,
`eta_2=0.4`, `eta_3=0.2`, `lambda_r=lambda_d=lambda_m=1.0`, QAA temperature `1.0`,
numerical epsilon `1e-8`, KNN size `50`, and inertia
range `[0.05, 0.5]`. CAR is applied in every communication round. Full uses the
exact gate `g=Ud*(1-Um)`, and the local objective contains only graph
reconstruction plus reliability-weighted global alignment; no standalone
uncertainty, hard-label, frequency-balancing, or cluster-balance loss is added.
`Um` combines global-target entropy, fixed-round aligned local/global JSD, and
original-feature proxy-neighborhood structural inconsistency. The Hungarian
permutation is estimated once per communication round and held fixed throughout
that round, as specified in the method. Local Student-t centers are initialized
by K-means on each pretrained client embedding. The inertia bounds are exposed
as `--zeta-min` and `--zeta-max` (defaults `0.05` and `0.5`).
Each `use_*` switch can also be overridden independently with its
`--use-...` or `--no-use-...` command-line form.

### Paper reliability and structure ablations

The paper-table ablations are orthogonal to `--experiment-mode full` and use
`--paper-ablation`:

- `qaa_wo_ud`, `qaa_wo_um`, and `qaa_wo_r` remove only the named term from
  the reliability score while retaining the complete Full client algorithm.
- `average_aggregation` uses uniform client weights in every round.
- `wo_proxy_initialization` replaces proxy-neighborhood initialization with
  an observed-view mean initializer; the hidden missing-view values are never read.
- `wo_semantic_refinement` disables iterative CAR semantic refinement.

`Full` and `average_aggregation` are shared by the reliability-component table
and the renamed structure-ablation table, so the two tables require seven unique
runs rather than nine. On AutoDL, run all seven BDGP 50%-missing experiments and
export both final-metric tables with:

```bash
bash run_paper_ablations_autodl.sh
```

The summary workbook is written to
`outputs/BDGP_4view_0.5_paper_ablations.xlsx`.

For the targeted ablation experiment, `no_uncertainty`, `ud_only`, `um_only`,
and `full` all measure both uncertainty values on every communication round.
Their training effects differ as follows:

- `no_uncertainty`: measures both values for diagnostics but does not use either
  uncertainty in CAR, weighted alignment, or QAA.
- `ud_only`: uses `Ud` for CAR and the client-level `Ud` term for QAA.
- `um_only`: uses `1-Um` for the controlled CAR ablation, uncertainty-weighted
  local alignment, and the client-level `Um` term for QAA.
- `full`: uses the paper gate `Ud*(1-Um)`, uncertainty-weighted alignment, and
  both uncertainty terms in QAA.

All four modes train on the same full imputed sample set. The method is active
from round 1; the first five rounds in plots are only a normalization reference.
QAA uses exact cross-client min-max normalization and softmax weights without
temporal EMA. Semantic recovery is performed at the end of each round and
recursively refines the previous recovered feature for the next round.
Synchronized post-local values are used for QAA and exported comparison curves.
ACC/NMI/ARI/F1 are measured from server KMeans on the quality-weighted fused
representation. Client-mean and unweighted ensemble scores remain in the log as
diagnostics.
Run the eight fixed 50%-missing ablation experiments with:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_eight_uncertainty_experiments.ps1
```

After the four modes of one dataset finish, create one three-panel comparison
figure (Ud, Um, and ACC) plus one comparison workbook with:

```bash
python compare_uncertainty_modes.py --name BDGP_4view --missrates 0.5 0.5 0.5 0.5
python compare_uncertainty_modes.py --name Caltech101-7 --missrates 0.5 0.5 0.5 0.5 0.5 0.5
```

The comparison command refuses to combine runs whose warm-up histories differ,
so an old or mismatched workbook cannot silently enter the final figure.

## Requirement

Tested experiment environment:

```bash
python==3.12.3
torch==2.3.0+cu121
cuda==12.1
```

Required Python packages used by this repository:

```bash
numpy
scipy
scikit-learn
h5py
openpyxl
matplotlib
munkres
cuml (RAPIDS, matching the installed CUDA major version)
```

For a CUDA 12 AutoDL environment, install and verify GPU KMeans with:

```bash
python -m pip install -r requirements-gpu-cu12.txt
python -c "from cuml.cluster import KMeans; print('cuML GPU KMeans ready')"
```
