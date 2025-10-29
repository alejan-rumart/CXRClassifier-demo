#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="cxr-demo"

echo "Creating conda env ${ENV_NAME} (Python 3.10)..."
conda create -n "${ENV_NAME}" python=3.10 -y

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Installing base requirements..."
pip install -r requirements.txt

echo "Installing Keras 3 (no deps) to avoid TF 2.15 resolver conflict..."
pip install --no-deps keras==3.3.3

echo "Done."
echo "To run:"
echo "  conda activate ${ENV_NAME}"
echo "  streamlit run app/streamlit_app.py"
