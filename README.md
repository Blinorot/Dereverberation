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

## Authors

- Petr Grinberg, EPFL

- Giuditta Del Sarto, EPFL
