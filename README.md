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
munkres
```
