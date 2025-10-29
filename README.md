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

## License

This repository is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

### Models & weights
- All code in this repository is MIT-licensed.
- Model weights (`*.keras`) and any example images may be subject to their own licenses or dataset terms. Ensure you have the right to distribute and use them. If you redistribute this repo without weights, keep Git LFS pointers but remove the actual files if required by their terms.

### Third-party notices
This project depends on third-party libraries (TensorFlow, PyTorch, torchxrayvision, pydicom, etc.) each under their own licenses. See their respective repositories for details.
