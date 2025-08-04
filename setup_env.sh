#!/usr/bin/env bash
set -e

python_version="$(cat .python-version)"

# 1. Install the interpreter if it’s missing
pyenv install -s "${python_version}"

# select python version for current shell
pyenv shell "${python_version}"

# create venv if missing
if [[ ! -d venv ]]; then
  python -m venv venv
fi

# 3. Activate venv & install packages
source venv/bin/activate

pip install -U pip setuptools wheel
pip install -r requirements.txt
