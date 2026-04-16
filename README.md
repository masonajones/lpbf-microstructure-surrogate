# LPBF Microstructure Surrogate

## Overview

This repository contains code for training, running, and analyzing a machine-learned surrogate model for predicting microstructure statistics in the Laser Powder Bed Fusion (LPBF) additive manufacturing process.

The surrogate is designed to accelerate localized predictions of grain size distribution statistics using a compact set of thermal characteristics extracted from LPBF simulations. The workflow couples simulation-derived thermal inputs with a PyTorch-based surrogate model to approximate microstructure outcomes much more quickly than running full microstructure simulation ensembles for every new condition.

This code was developed as part of Mason Jones's PhD research at the University of California, Davis, in collaboration with Sandia National Laboratories.

## Repository Contents

Core files in this repository include:

- `PrancingPony_ADnlgP90b256_final.pt` — pre-trained surrogate model weights for inference.
- `SPKML.yml` — Conda environment specification for the Python dependencies used by the project.
- `data_points_metadata.csv` — metadata table used to indicate which thermal and microstructure data files to load.
- `microstructure_surrogate_train.py` — main multi-GPU training script using PyTorch Distributed Data Parallel and TensorDict memmaps.
- `microstructure_surrogate_DDP_tensordict_optim.py` — distributed hyperparameter-optimization workflow under active development.
- `microstructure_surrogate_optim.py` — earlier hyperparameter-optimization workflow.
- `microstructure_surrogate_infpltcomp.py` — inference and accuracy measurement script.
- `microstructure_surrogate_sensitivity.py` — Jacobian-based sensitivity statistics of the model outputs with respect to the thermal input cubes.
- `thermal_analysis.py` — script for analyzing and plotting the thermal characteristics used as model inputs.

## Installation

Clone the repository and create the Conda environment:


```bash
git clone https://github.com/masonajones/lpbf-microstructure-surrogate.git
cd lpbf-microstructure-surrogate

conda env create -f SPKML.yml
conda activate SPKML
```

The environment file includes the main Python dependencies used by the repository, including PyTorch, TensorDict, CuPy, Optuna, meshio, and plotting and data-analysis packages.

## Hardware and Runtime Notes

These scripts were developed primarily for GPU-equipped Linux and HPC environments.

- Training is intended for CUDA-enabled systems and is configured for multi-GPU execution using Distributed Data Parallel in `microstructure_surrogate_train.py`.
- Several scripts hard-code CUDA device indices such as `cuda:2` or `cuda:7` when GPUs are available. On systems with a different GPU layout, these values may need to be edited before use.
- Some scripts also assume access to a local scratch directory such as `/scratch/<user>/` for lower-latency data access.
- The training workflow is not currently set up as a robust CPU-only path.
- Inference may run on CPU in principle, but practical use is still oriented toward GPU systems.

For best performance, place the dataset and TensorDict memmaps on a fast local SSD or scratch filesystem rather than a network-mounted drive.

## Data Availability

The full training dataset is not bundled with this repository, but may be made available upon request.

The full workflows and simulation codes used for data generation are not yet available.

To train or run comparison workflows, you will need the thermal simulation outputs and the corresponding microstructure histogram data indicated by `data_points_metadata.csv`. The scripts assume a directory structure built around the folders:

- `thermal_data/`
- `MS_data/`
- optional scratch-backed TensorDict storage such as `/scratch/<user>/data_tensordict`

The exact files loaded are determined from `data_points_metadata.csv` together with the configuration variables near the top of each script.

## Example Directory Layout

A typical working layout might look like this:

```text
lpbf-microstructure-surrogate/
├── PrancingPony_ADnlgP90b256_final.pt
├── SPKML.yml
├── data_points_metadata.csv
├── microstructure_surrogate_train.py
├── microstructure_surrogate_infpltcomp.py
├── microstructure_surrogate_sensitivity.py
├── thermal_analysis.py
├── ...
├── thermal_data/
│   ├── L1P90V105.xmf
│   ├── L1P124V109_L1.xmf
│   └── ...
└── MS_data/
    ├── n200/
    │   ├── L1P90V105_hist1_n200.npy
    │   └── ...
    ├── n100/
    └── ...
```

If you use scratch-backed memmaps, you may also have a separate location such as:

```text
/scratch/<user>/
├── MS_data/
└── data_tensordict/
```

In practice, `thermal_data/` may remain in the repository working directory while `MS_data/` and `data_tensordict/` may instead live under `/scratch/<user>/`, depending on `use_scratch`, `scratch_dir`, and the script you are running.

## Configuration Pattern

The scripts in this repository are primarily configured by editing variables near the top of each Python file rather than by passing a formal command-line interface.

Important settings commonly include:

- `Model_name`
- `metadata_filename` or `metadata_file`
- `Training`
- `Testing`
- `ensemble_size`
- `ensemble_treatment`
- `use_scratch`
- `scratch_dir`
- device selection such as `cuda:2`

For most users, the first settings likely to need adjustment are:

- `use_scratch`
- `scratch_dir`
- the CUDA device selection used by the script

The usual workflow is therefore:

1. Open the target script.
2. Edit the configuration block near the top.
3. Run the script.

## Typical Workflows

### 1. Train a New Surrogate Model

The main training entry point is:

```bash
python microstructure_surrogate_train.py
```

This script:

- reads case definitions from `data_points_metadata.csv`
- constructs a TensorDict-backed dataset
- splits melted voxels into training and test sets
- trains a 3D ConvNeXt-style surrogate model
- saves model checkpoints and training and test logs

The training script is configured for distributed multi-GPU execution and uses `torch.multiprocessing.spawn` together with DDP.

### 2. Run Inference with the Pretrained Model

To generate predictions with the included checkpoint, and optionally compare them to simulation-based reference data:

```bash
python microstructure_surrogate_infpltcomp.py
```

By default, this script is configured to use:

- `metadata_filename = "data_points_metadata.csv"`
- `Model_name = "PrancingPony_ADnlgP90b256"`

Comparison and plotting are controlled explicitly inside the script.

- Setting `Compare_and_Plot = True` enables comparison and plotting.
- When comparison is enabled, the script expects the corresponding reference microstructure histogram files to be available.
- The script does not automatically fall back to prediction-only mode if those reference files are missing.
- Setting `Compare_and_Plot = False` disables the comparison path, but fully prediction-only use may still require minor script edits because the current implementation remains oriented around workflows that have reference microstructure data available.

Before running this script on a new system, you will likely need to review `use_scratch`, `scratch_dir`, and the hard-coded CUDA device selection.

### 3. Run Sensitivity Analysis

To compute Jacobian-based sensitivity statistics for the pretrained model:

```bash
python microstructure_surrogate_sensitivity.py
```

This script computes summary statistics of output sensitivity with respect to the thermal input cubes and saves NumPy arrays containing the resulting quantities.

### 4. Analyze Thermal Inputs

To analyze the thermal characteristics used as surrogate inputs:

```bash
python thermal_analysis.py
```

This script can be used to inspect distributions and relationships in the thermal input space across the available cases.

### 5. Explore Hyperparameter Optimization Workflows

Two optimization scripts are included:

```bash
python microstructure_surrogate_optim.py
python microstructure_surrogate_DDP_tensordict_optim.py
```

These scripts reflect experimental or work-in-progress optimization workflows. They are useful as references, but the primary supported training path in the repository is still `microstructure_surrogate_train.py`.

## Current Limitations

A few practical limitations are worth noting up front:

- The repository does not currently provide a polished command-line interface.
- Most runtime settings are hard-coded in the scripts.
- File paths and scratch-directory assumptions may need adjustment on a new machine.
- Some optimization and analysis scripts remain research-grade rather than packaged end-user tools.
- Training and comparison workflows depend on external datasets not included in the repository.
- The current inference script is still oriented around workflows where reference microstructure data are available; prediction-only use without those references may require minor script edits.

## Citation

If you use this code or model in academic work, please cite the associated dissertation:

> Mason Jones, *A Machine Learning Approach to Accelerated Prediction of Statistics of Microstructures Produced by the Laser Powder Bed Fusion Process*, PhD Dissertation, University of California, Davis, 2026.

## Acknowledgements

This work was developed in the context of research conducted at the University of California, Davis, in collaboration with Sandia National Laboratories.
