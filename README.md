# Dereverberation

This repository contains dereverberation algorithms written in Python.

We implemeted several approaches, including [HERB](https://ieeexplore.ieee.org/abstract/document/4032782) and [LP Residual](https://ieeexplore.ieee.org/abstract/document/1621193).

## Installation

To install our repository, follow these steps:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To perform dereverberation on one of the datasets from `src/datasets`, use the following command:

```bash
python3 dereverberate.py -d=DATASET_NAME -a=ALGORITHM_NAME
```

Where `DATASET_NAME` is name of the class from `src/datasets` and `ALGORITHM_NAME` is the name of the dereverberation algorithm (HERB, LP, WPE). Dereverberated signals are saved in `data/dereverberated/DATASET_NAME`.

To calculate metrics after dereverberation, run this script:

```bash
python3 scripts/calculate_metrics.py -d=DATASET_NAME
```

Metrics dictionary will be saved in `data/DATASET_NAME_metrics.pth`.

## Authors

- Petr Grinberg, EPFL

- Giuditta Del Sarto, EPFL
