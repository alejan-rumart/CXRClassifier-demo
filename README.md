# Chest X-ray Demo – MSc Thesis (Alejandro Ruiz Martínez)

A local Streamlit app to run CNN inference on chest X-rays, view per-label probabilities,
apply operating thresholds, and visualize Grad-CAM heatmaps. Supports DICOM and PNG/JPG inputs.

## Quickstart
```bash
# 1) Create env (example with conda)
conda create -n cxr-demo python=3.10 -y
conda activate cxr-demo

# 2) Install base requirements
pip install -r requirements.txt

# 3) Install Keras 3 separately (avoids TF 2.15 resolver conflict)
pip install --no-deps keras==3.3.3

# 4) Run
streamlit run app/streamlit_app.py
```
## Install (Windows/macOS/Linux)

Requires Conda/Miniconda and Git LFS (for large model files).

Fast path (Windows)
```
install.bat
```
Fast path (macOS / Linux)
```
./install.sh
```
Manual
```
conda create -n cxr-demo python=3.10 -y
conda activate cxr-demo
pip install -r requirements.txt
# IMPORTANT: install Keras 3 separately to avoid TF 2.15 resolver conflict
pip install --no-deps keras==3.3.3
```