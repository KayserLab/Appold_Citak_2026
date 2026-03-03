# Appold_Citak_2026

`Appold_Citak_2026` contains the code needed to recreate the simulations, analysis, and figure panels for the paper [Spatial resource dynamics control resistance escpae](<https://doi.org/10.64898/2025.12.22.695823>).

- [Overview](#Overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#Demo)
- [General Use](#general-use)

## Overview

The core model (`source/core.py`) of this project is a 2D diffusion-based colony simulation with sensitive and resistant cell populations, nutrient diffusion/uptake, treatment on/off schedules with delayed efficacy, and stochastic mutation events.

### Repository Structure

- `source/`: simulation engine, runners, fitting, visualization.
- `Figure_1` ... `Figure_5`: panel generation scripts for the main figures.
- `SI_Figures/`: supplementary figure scripts.
- `Experimental_analysis/`: experimental data processing scripts.
- `params.yaml`: simulation parameters.

The code expects the following data folder structure:
- `data/`:
  - `sim_data`
  - `exp_data`
  - `sweeps`
  - `sweep_arrays`

## System Requirements

### Software

#### OS Requirements

The code was run and tested on Windows 11 (Version 24H2, Build 26100.7623)

#### Python requirements
- Python `3.11`. (The package `aicsimageio` requires python 3.11, if only code without this package is run (e.g. simulation code) newer python versions are also supported)
- Python packages in `requirements.txt`:
  - `matplotlib`, `numpy`, `pandas`, `pyyaml`, `scikit-image`, `scipy`, `torch`, `tqdm`, `dask`, `h5py`, `opencv-python`, `seaborn`, `statsmodels`, `trackpy`, `aicsimageio`.
- Optional for video export: `ffmpeg` (required by `matplotlib.animation.FFMpegWriter`).

### Hardware
It is recommended to use a computer with 16+ GB RAM, 100+ GB free disk, and a multi-core CPU for full simulation/figure reproduction, but for smaller demo runs a standard computer is sufficient. 

Notes:
- Full simulation arrays are large. A single run can create multiple `.npy` files each in the hundreds of MB.
- Batch runs in `data/sim_data/` can consume substantial disk space.
- The parameter sweeps can take significant time when run on a low core count CPU

## Installation Guide

```bash
git clone https://github.com/KayserLab/Appold_Citak_2026.git
cd Appold_Citak_2026
pip install -e .
```

The expected install time - including the download of all required packages - is 5 minutes.

## Demo 

The expected runtime for the entire demo is dependend on the hardware up to a few minutes.

### Simulation Demo
To run a demo of the simulation run the `demo.py` file in the `demo` folder. It will create 2 simulation for the 6.5h/18h treatment schedule. The expected outcome for this file are in the `demo/demo_data/met_6_5_18` folder:
- `met_6_5_18_0/`:
  - `nutrients.npy`
  - `sensitive.npy`
  - `resistant.npy`
  - `treatment_times.npy`
  - `treatment_efficacy.npy`
  - `params.pth`
- `met_6_5_18_1/`:
  - `nutrients.npy`
  - `sensitive.npy`
  - `resistant.npy`
  - `treatment_times.npy`
  - `treatment_efficacy.npy`
  - `params.pth`

### Figure Plots Demo
In the folder `demo/demo_figures` are the scripts `figure_4_sweep_demo.py` and `figure_4_trajectories_demo.py` which will use the provided data in the `demo/demo_data` folder to create example plots of the trajectories and sweeps of the treatment schedules 4h/18h, 6.5h/18h, and 9h/18h.

The `kymo_demo.py` file will create a kymograph of the 6.5h/18h treatment schedule. Important: It is necessary to run the `demo.py` file before the create the needed data!

The figure plots demo files need to be run from the root directory.

### Simulation Movie

To create a video of the created simulation change the path in the main function of `source/visualization/animate_colonies.py` to
```bash
path = 'demo/demo_data/met_6_5_18/met_6_5_18_0'
```
and run the script from the root directory. The created video will be saved in the videos folder. Important: It is necessary to run the `demo.py` file before the create the needed data! (Note: ffmpeg is required to create videos)

## General Use

Dependent on the IDE it may be needed to update some paths in this repository to run all scripts without errors.

### 1. Create simulation data used in the paper

Run the `source/create_simulation_data.py` script:

Default output path pattern:
- `data/sim_data/<treatment>/<treatment>_<replicate>/`

### 2. Create parameter sweeps

Run `source/executable.py` with the wanted sweep parameters set in `params.yaml` to create the sweep data. Then run the `Figure_4/panel_a_c/create_sweep_arrays.py` to get the analysed sweeps arrays, which are used by other scripts.

### 3. Generate figures

Figure scripts are organized in `Figure_1` ... `Figure_5` and `SI_Figures`.

After creating the simulation data in step 1 and putting the experimental data in the corresponding folder (see [Overview](#overview)) the panel scripts can be run to create the figure panels.

### 4. Create animations

Run `source/visualization/animate_colonies.py` with the path changed to the wanted simulation data to create animations of the run. The options `plot_nutes` allows you to choose if just the colony is animated or to additionally animate the nutrient layer. 

This script requires `ffmpeg`.

### 5. Run parameter fitting

Important:
- Fitting scripts expect specific experimental data folders in the fit folder: `source/fit/fit_data/no_treatemnt_csv/...`, `source/fit/fit_data/mutation_rate/...` (this should be continuous dose data).
- Ensure the folder `fit` contains the following subfolder:
  - `logs_fitting`
  - `fit_results`
  - `fit_data`

Fit in the following order, always adjusting the parameters to the newest value:
- `source/fit/parameter_fitting/fit_disperion_and_nutrients.py`
- `source/fit/parameter_fitting/fit_mutation_rate.py`
- `source/fit/parameter_fitting/fit_mutation_scaling.py`

Note: The growth rates are calculated directly from plate reader experiments in `source/fit/parameter_fitting/calc_growth_rates.py`.

### Changing simulation parameters

To change the simulation paramters edit `params.yaml` at the repository root.
