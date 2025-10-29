# app/pages/Instructions.py
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Instructions – Chest X-ray Demo", page_icon="📘", layout="wide")

st.title("Instructions & User Guide")
st.caption("This demo is for single-image experimentation. For correct interpretation, please read the thesis.")

st.markdown("---")

# =========================
# Overview
# =========================
st.header("What this app does")
st.markdown(
"""
This application lets you:
- **Upload** up to 5 chest PA radiographs (`PNG/JPG/DICOM`).
- **Preprocess** each image exactly as in the thesis (lung-mask crop when available, otherwise a safe fallback; final size `224×224`).
- **Run inference** with one or more CNN models trained for multilabel findings.
- **Optionally ensemble** multiple models per image.
- **Optionally view Grad-CAM** heatmaps for a selected image/model/label. The Grad-CAM heatmap is intended to explain why or how the model took a specific decision in assigning some score in a label to an image. It should be use to check that the model correctly looked at the proper region when it predicted a high score for a label
"""
)

st.warning("This demo is **not** a medical device and must **not** be used for clinical decisions. Please read the thesis for the proper interpretation of results.")

st.markdown("---")

# =========================
# Typical workflow
# =========================
st.header("Workflow")

st.subheader("1) Upload images")
st.markdown(
"""
- Use **“Upload chest PA radiographs (PNG/JPG/DICOM) – up to 5”**.
- Supported formats: `*.png`, `*.jpg`/`*.jpeg`, and DICOM `*.dcm`.
- For DICOM, a mini-header shows basics (e.g., **ViewPosition**).  
- After uploading, click **Run preprocessing**.  
  You’ll see:
  - **Original**: the image as read (DICOM is windowed and normalized).
  - **Preprocessed (224×224)**: the input the CNN actually receives.
  - A **Crop** tag tells you if **lung-mask** (preferred) or **fallback** cropping was used.
"""
)

st.subheader("2) Pick models & thresholds (sidebar)")
st.markdown(
"""
- In **Models**, pick one or more models. The app automatically aligns them to the **common label set**.
- In **Per-label thresholds**:
  - Choose a **default** for all labels: **None** / **High Sens** / **High Spec**, then click **Apply default to all**.
  - Optionally override per label under **Set per-label operating point**.
- **High Sens** and **High Spec** thresholds come from each model’s `operating_points_table.csv` (placed inside each model folder).  
  These thresholds were derived in validation (see the thesis).
"""
)

st.subheader("3) Run inference")
st.markdown(
"""
- Click **Run inference** to populate the **Results** table.  
- The table structure is **grouped by image**, with **sub-columns for each model** and an **Ensemble** column:
  - **If a label’s threshold is “None”** → the cell shows a **probability** (0–1, 3 decimals).
  - **If a label’s threshold is High-Sens or High-Spec** → the cell shows **POS/NEG**.
- You can **download two CSVs**:
  - **Numeric probabilities** (always raw probabilities for all image×model cells plus Ensemble = mean probability),
  - **Displayed table** (matches exactly what is shown on screen: probabilities and/or POS/NEG).
"""
)

st.subheader("4) Grad-CAM")
st.markdown(
"""
- Pick **Image**, **Model**, and **Label**, then click **Generate Grad-CAM**.
- You’ll see:
  - The **preprocessed** image,
  - The **Grad-CAM** map (0–1),
  - The **overlay**.  
- Generating Grad-CAM **does not** clear the inference table.
"""
)

st.markdown("---")

# =========================
# Ensemble modes
# =========================
st.header("Ensemble modes")
st.markdown(
"""
Pick one under **Ensemble → Mode**:

- **Average**: shows the **mean probability** in the Ensemble column (used when a label’s threshold is **None**).
- **CTP (avg→threshold)**: for labels with thresholds, take the **mean probability** and compare it against the **mean threshold** across the selected models → **POS/NEG**.
- **PTC (vote)**: for labels with thresholds, convert each model’s probability to **POS/NEG** using its own threshold, then take a **majority vote** → **POS/NEG**.

When **threshold = None** for a label, the Ensemble column always shows the **mean probability**, regardless of the ensemble mode.
"""
)

st.markdown("---")

# =========================
# Interpreting thresholds
# =========================
st.header("Operating points")
st.markdown(
"""
Each model folder (e.g., `app/models/enb0_cutmix_best/`) should include:
- `model.keras` — the trained CNN,
- `run_config.json` — training details / label list,
- `operating_points_table.csv` — per-label thresholds (**High Sens** / **High Spec**).

**High Sens** → threshold selected so that sensitivity is ≳ target (e.g., 95%) on the reference split.  
**High Spec** → threshold selected so that specificity is ≳ target (e.g., 95%) on the reference split.

Exact procedure and caveats are documented in the thesis. Use these operating points as decision **aids**, not ground truth.
"""
)

st.markdown("---")

# =========================
# DICOM notes
# =========================
st.header("DICOM specifics")
st.markdown(
"""
- DICOMs are normalized to 8-bit for preview and preprocessed to `224×224` for the CNN.
- The app tries to handle common transfer syntaxes. If a file won’t open, convert it to `PNG`/`JPG` externally.
- If an image comes out too dark/bright, it usually means the DICOM stored unusual pixel ranges or metadata.  
  The app applies sensible defaults, but for some edge cases manual conversion may work better.
"""
)

st.markdown("---")

# =========================
# Troubleshooting
# =========================
st.header("Troubleshooting")
with st.expander("I see **Crop: fallback**"):
    st.markdown(
    """
    The lung segmentation mask wasn’t available/reliable. The app used a safe centered crop.  
    You can still run inference; just interpret Grad-CAM cautiously.
    """
    )
with st.expander("My image looks **all black / all white**"):
    st.markdown(
    """
    Usually a DICOM windowing/range issue. Try exporting to PNG/JPG with your PACS viewer and upload that file.
    """
    )
with st.expander("Why do model probabilities differ a lot?"):
    st.markdown(
    """
    Models were trained with **different recipes**. Probability calibration and operating points may vary.  
    Compare **ROC-AUC** per label in the **Models** page and see the thesis for details.
    """
    )
with st.expander("Where are the downloads?"):
    st.markdown(
    """
    After **Run inference**, scroll below the table. You’ll find:
    - **Download numeric probabilities (CSV)**
    - **Download displayed table (CSV)**
    """
    )

st.markdown("---")

# =========================
# Where files live
# =========================
st.header("Project layout (relevant bits) - 1st version")
st.code(
""".
├─ app/
│  ├─ streamlit_app.py               # main app
│  ├─ engine/
│  │  ├─ preprocessing.py            # image preprocessing and lung-mask crop
│  │  ├─ inference.py                # model loading & prediction
│  │  ├─ gradcam.py                  # Grad-CAM utilities
│  │  └─ dicom_io.py                 # DICOM reading & normalization
│  ├─ models/
│  │  ├─ enb0_cutmix_best/
│  │  │  ├─ model.keras
│  │  │  ├─ run_config.json
│  │  │  └─ operating_points_table.csv
│  │  └─ enb0_nomix_best/
│  │     ├─ model.keras
│  │     ├─ run_config.json
│  │     └─ operating_points_table.csv
│  └─ pages/
│     ├─ Models.py                   # model summaries (ROC-AUC table)
│     └─ Instructions.py             # (this page)
└─ requirements.txt
""",
    language="text",
)

st.markdown("---")

# =========================
# Privacy & credits
# =========================
st.header("Privacy")
st.markdown(
"""
- Processing and inference run **locally** on your machine.
- Uploaded files stay in memory/session cache for the app session.
- No data is sent to external servers by this app.
"""
)

st.header("Credits")
st.markdown(
"""
Created by **Alejandro Ruiz Martínez** · [Website](https://alejan-rumart.github.io/)
"""
)
