# app/streamlit_app.py
from __future__ import annotations
from pathlib import Path
from io import BytesIO
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from engine.preprocessing import preprocess_png_or_jpg
from engine.inference import (
    load_keras_model,
    predict_probs,
    discover_model_dirs,
    load_model_labels_from_dir,
    load_model_operating_points_from_dir,
)
from engine.dicom_io import is_dicom, load_dicom, UnsupportedCompressedDicomError
from engine.gradcam import get_or_build_grad_model, grad_cam_from_feat, overlay_heatmap

# ---------------- Page & paths ----------------
st.set_page_config(page_title="Chest X-ray Demo – MSc Thesis (AR Martínez)", page_icon="🫁", layout="wide")
APP_ROOT = Path(__file__).parent
MODELS_ROOT = APP_ROOT / "models"
MAX_FILES = 5
THUMB_W = 220

# Hide Streamlit toolbar/menu/footer
st.markdown("""
<style>
[data-testid="stToolbar"] {display: none !important;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none !important;}
</style>
""", unsafe_allow_html=True)

# ---------------- Session init ----------------
ss = st.session_state
ss.setdefault("preproc_cache", {})      # key -> {name, is_dicom, meta, viz_orig, viz_crop224, x224, used_mask}
ss.setdefault("per_label_op", {})       # label(lower) -> "None"/"High Sens"/"High Spec"
ss.setdefault("inference_table", None)  # persisted table
ss.setdefault("labels_ref", [])         # last known label list (lowercase)
ss.setdefault("preproc_done", False)

# ---------------- Sidebar: Models ----------------
st.sidebar.header("Models")
discovered: Dict[str, Path] = discover_model_dirs(MODELS_ROOT)
if not discovered:
    st.sidebar.warning("No models found under app/models/.")
model_names = list(discovered.keys())
default_pick = model_names[:2] if model_names else []
picked_names = st.sidebar.multiselect("Select models", options=model_names, default=default_pick)

@st.cache_resource(show_spinner=False)
def _load_model_from_dir(model_dir: Path, model_name: str):
    labels = load_model_labels_from_dir(model_dir)
    ops = load_model_operating_points_from_dir(model_dir)  # may be None
    # NEW: warn if CSV exists but parsing failed
    csvp = model_dir / "operating_points_table.csv"
    if ops is None and csvp.exists():
        st.sidebar.warning(
            f"Operating points file found for **{model_name}** "
            f"but couldn’t be parsed. Thresholds for this model are disabled."
        )
    model = load_keras_model(model_dir / "model.keras")
    return model, labels, ops


selected_models: Dict[str, Tuple] = {
    name: _load_model_from_dir(discovered[name], name) for name in picked_names
}
# Common labels across selected models
def _intersect_labels(models: Dict[str, Tuple]) -> List[str]:
    if not models:
        return []
    sets = []
    for _, (_, labels, _) in models.items():
        sets.append(set([l.lower() for l in labels]))
    common = set.intersection(*sets) if sets else set()
    return sorted(common)

ref_labels = _intersect_labels(selected_models)
if ref_labels != ss["labels_ref"]:
    ss["labels_ref"] = ref_labels
    ss["per_label_op"] = {lbl: ss["per_label_op"].get(lbl, "None") for lbl in ref_labels}

# ---------------- Sidebar: Ensemble & thresholds ----------------
st.sidebar.subheader("Ensemble")
ens_choice = st.sidebar.radio(
    "Mode",
    ["Average", "CTP (avg→threshold)", "PTC (vote)"],
    index=0,
    help="For labels with 'None' threshold, ensemble uses Average. For labels with thresholds, choose CTP or PTC."
)

st.sidebar.subheader("Per-label thresholds")
default_op = st.sidebar.selectbox("Default for all labels", ["None", "High Sens", "High Spec"], index=0)
if st.sidebar.button("Apply default to all"):
    for k in ref_labels:
        ss["per_label_op"][k] = default_op

if ref_labels:
    with st.sidebar.expander("Set per-label operating point", expanded=False):
        for lbl in ref_labels:
            ss["per_label_op"][lbl] = st.selectbox(
                lbl, ["None", "High Sens", "High Spec"],
                index=["None","High Sens","High Spec"].index(ss["per_label_op"][lbl]),
                key=f"op_{lbl}"
            )

# ---------------- Title & notice ----------------
st.title("Chest X-ray Classifier Demo (MSc Thesis)")
st.info("This demo is for single-image experimentation. For correct interpretation, **please read the thesis**.")

# ---------------- Step 1: Upload ----------------
st.header("1) Upload chest PA radiographs (PNG/JPG/DICOM) – up to 5")

uploaded_files = st.file_uploader(
    "Drop files here or browse",
    type=["png", "jpg", "jpeg", "dcm"],
    accept_multiple_files=True,
)

if uploaded_files and len(uploaded_files) > MAX_FILES:
    st.warning(f"Only the first {MAX_FILES} files will be used.")
    uploaded_files = uploaded_files[:MAX_FILES]

def _load_as_pil_or_dicom(file):
    """Return (PIL image for preview, meta, is_dicom). Raises on fatal errors."""
    name = getattr(file, "name", "")
    suffix = Path(name).suffix.lower()

    # 1) If the user clearly gave an image file, open it as such (avoid DICOM probe)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        try:
            return Image.open(BytesIO(file.getvalue())), {}, False
        except Exception as e:
            st.warning(f"**{name}**: Could not open as PNG/JPG. {e}")
            raise

    # 2) Otherwise probe for DICOM first, then try image fallback
    buf = BytesIO(file.getvalue())
    if is_dicom(buf):
        try:
            preview_u8, _, meta = load_dicom(buf)
            return Image.fromarray(preview_u8), meta, True
        except UnsupportedCompressedDicomError as e:
            st.warning(f"**{name}**: {e}")
            raise
        except Exception as e:
            st.warning(f"**{name}**: Could not read DICOM file. {e}")
            raise
    else:
        try:
            return Image.open(BytesIO(file.getvalue())), {}, False
        except Exception as e:
            st.warning(f"**{name}**: Could not open as PNG/JPG. {e}")
            raise

# Manual preprocessing
run_pp = st.button("Run preprocessing", type="primary")

current_keys = []
if uploaded_files:
    for f in uploaded_files:
        current_keys.append(f"{f.name}:{f.size}")

# Drop stale cache entries if files changed
stale = set(ss["preproc_cache"].keys()) - set(current_keys)
for k in stale:
    ss["preproc_cache"].pop(k, None)
if stale:
    ss["preproc_done"] = False

# Run preprocessing on demand
if run_pp and uploaded_files:
    processed_any = False
    for f in uploaded_files:
        key = f"{f.name}:{f.size}"
        if key in ss["preproc_cache"]:
            continue

        # 1) Try to load as DICOM or PNG/JPG; warn & skip on failure
        try:
            pil_img, meta, is_dcmf = _load_as_pil_or_dicom(f)
        except Exception as e:
            st.warning(f"Skipping **{f.name}** due to read error: {e}")
            continue

        # 2) Run your preprocessing; warn & skip on failure
        try:
            x224, viz_psp, viz_crop224, viz_orig, used_mask = preprocess_png_or_jpg(pil_img)
        except Exception as e:
            st.warning(f"Skipping **{f.name}** due to preprocessing error: {e}")
            continue

        # 3) Cache successful result
        ss["preproc_cache"][key] = {
            "name": f.name,
            "is_dicom": is_dcmf,
            "meta": meta,
            "viz_orig": viz_orig,
            "viz_crop224": viz_crop224,
            "x224": x224,
            "used_mask": bool(used_mask),
        }
        processed_any = True

    # Mark as done only if we have at least one cached item
    ss["preproc_done"] = processed_any or bool(ss["preproc_cache"])


# Previews
if ss["preproc_done"] and ss["preproc_cache"]:
    for key, pack in ss["preproc_cache"].items():
        colL, colR = st.columns(2)
        if pack["is_dicom"]:
            info = pack["meta"]
            hdr = f"DICOM • Size: {pack['viz_orig'].shape[1]}×{pack['viz_orig'].shape[0]} • View: {info.get('ViewPosition','') or 'N/A'} • Crop: {'lung-mask' if pack['used_mask'] else 'fallback'}"
        else:
            hdr = f"PNG/JPG • Size: {pack['viz_orig'].shape[1]}×{pack['viz_orig'].shape[0]} • Crop: {'lung-mask' if pack['used_mask'] else 'fallback'}"
        st.markdown(f"**{pack['name']}** — {hdr}")
        with colL:
            st.image(pack["viz_orig"], caption="Original", width=THUMB_W, clamp=True)
        with colR:
            st.image(pack["viz_crop224"], caption="Preprocessed (224×224)", width=THUMB_W, clamp=True)
else:
    if uploaded_files:
        st.caption("Press **Run preprocessing** to prepare images.")

# ---------------- Step 2: Inference ----------------
st.header("2) Run inference")

def _apply_thresh_for_model(prob: float, label: str, ops: dict | None, choice: str) -> Tuple[str, bool]:
    if choice == "None" or ops is None:
        return f"{prob:.4f}", False
    key = "sens" if choice == "High Sens" else "spec"
    thr_map = ops.get(key) if ops else None
    if not thr_map:
        return f"{prob:.4f}", False
    thr = thr_map.get(label.lower())
    if thr is None or not np.isfinite(thr):
        return f"{prob:.4f}", False
    yhat = int(prob >= float(thr))
    return ("POS" if yhat else "NEG"), True

def _ensemble_cell(label: str, label_choice: str, per_model_probs: List[float], ops_list: List[dict|None]) -> str:
    if label_choice == "None":
        if not per_model_probs:
            return "N/A"
        return f"{float(np.mean(per_model_probs)):.4f}"
    k = "sens" if label_choice == "High Sens" else "spec"
    thrs = []
    for ops in ops_list:
        if ops is None:
            return "N/A"
        t = (ops.get(k) or {}).get(label.lower())
        if t is None or not np.isfinite(t):
            return "N/A"
        thrs.append(float(t))
    if ens_choice.startswith("CTP"):
        avg_score = float(np.mean(per_model_probs)) if per_model_probs else np.nan
        if not np.isfinite(avg_score):
            return "N/A"
        avg_thr = float(np.mean(thrs))
        return "POS" if avg_score >= avg_thr else "NEG"
    elif ens_choice.startswith("PTC"):
        votes = [1 if p >= t else 0 for p, t in zip(per_model_probs, thrs)]
        return "POS" if (np.sum(votes) >= (len(votes) + 1) // 2) else "NEG"
    else:
        avg_score = float(np.mean(per_model_probs)) if per_model_probs else np.nan
        if not np.isfinite(avg_score):
            return "N/A"
        avg_thr = float(np.mean(thrs))
        return "POS" if avg_score >= avg_thr else "NEG"

run_inf = st.button("Run inference", type="primary")

if run_inf:
    if not ss["preproc_done"] or not ss["preproc_cache"]:
        st.warning("Preprocess the images first.")
    elif not selected_models:
        st.warning("Select at least one model.")
    elif not ref_labels:
        st.warning("No common labels across selected models.")
    else:
        # 1) MultiIndex columns as (image ▶ model)
        img_keys = list(ss["preproc_cache"].keys())
        img_names = [ss["preproc_cache"][k]["name"] for k in img_keys]

        cols = []
        for img in img_names:
            for mn in selected_models.keys():
                cols.append((img, mn))
            cols.append((img, "Ensemble"))
        columns = pd.MultiIndex.from_tuples(cols, names=["image", "model"])

        # 2) Two tables: display (strings) and numeric probabilities (floats)
        idx = pd.Index(ref_labels, name="label")
        table_display = pd.DataFrame(index=idx, columns=columns, dtype=object)
        table_probs   = pd.DataFrame(index=idx, columns=columns, dtype=float)

        # 3) Per-image inference
        for img_key, img_name in zip(img_keys, img_names):
            pack = ss["preproc_cache"][img_key]
            x = pack["x224"]

            # Collect per-model probabilities as a dict: model -> list[float] aligned to ref_labels
            per_model_probs_map: Dict[str, List[float]] = {}
            ops_per_model: List[dict | None] = []

            for model_name, (model, labels, ops) in selected_models.items():
                probs = predict_probs(model, x)  # shape (n_labels_model,)
                lbl_map = {labels[i].lower(): float(probs[i]) for i in range(len(labels))}
                per_model_probs_map[model_name] = [lbl_map.get(lbl, np.nan) for lbl in ref_labels]
                ops_per_model.append(ops)

            # Fill individual model columns
            for model_name, (_, _, ops) in selected_models.items():
                probs_vec = per_model_probs_map[model_name]
                for lbl, p in zip(ref_labels, probs_vec):
                    # numeric
                    table_probs.loc[lbl, (img_name, model_name)] = p
                    # displayed (prob or POS/NEG depending on per-label OP)
                    disp, _ = _apply_thresh_for_model(p, lbl, ops, ss["per_label_op"].get(lbl, "None"))
                    table_display.loc[lbl, (img_name, model_name)] = disp

            # Fill Ensemble column under this image
            for i, lbl in enumerate(ref_labels):
                probs_for_lbl = [per_model_probs_map[mn][i] for mn in selected_models.keys()]
                # numeric ensemble = mean probability (ignores thresholding)
                mean_p = float(np.nanmean(probs_for_lbl)) if np.isfinite(np.nanmean(probs_for_lbl)) else np.nan
                table_probs.loc[lbl, (img_name, "Ensemble")] = mean_p

                # displayed ensemble respects per-label OP and chosen ensemble mode
                cell = _ensemble_cell(
                    lbl,
                    ss["per_label_op"].get(lbl, "None"),
                    probs_for_lbl,
                    [selected_models[mn][2] for mn in selected_models.keys()],
                )
                table_display.loc[lbl, (img_name, "Ensemble")] = cell

        # Persist both
        ss["inference_table"] = table_display
        ss["inference_probs"] = table_probs


if ss.get("inference_table") is not None:
    st.subheader("Results (rows = labels, columns = images ▶ models)")
    st.dataframe(ss["inference_table"], use_container_width=True)

    # Downloads: numeric probabilities and the shown table
    c1, c2 = st.columns(2)
    with c1:
        if ss.get("inference_probs") is not None:
            st.download_button(
                "Download numeric probabilities (CSV)",
                ss["inference_probs"].to_csv(index=True).encode("utf-8"),
                file_name="predictions_numeric_models_by_image.csv",
                mime="text/csv",
            )
    with c2:
        st.download_button(
            "Download displayed table (CSV)",
            ss["inference_table"].to_csv(index=True).encode("utf-8"),
            file_name="predictions_display_models_by_image.csv",
            mime="text/csv",
        )


# ---------------- Step 3: Grad-CAM ----------------
st.header("3) Grad-CAM")

if ss["preproc_done"] and ss["preproc_cache"] and selected_models:
    img_options = list(ss["preproc_cache"].keys())

    c1, c2, c3 = st.columns(3)
    with c1:
        pick_img_key = st.selectbox("Image", options=img_options, format_func=lambda k: ss["preproc_cache"][k]["name"])
    with c2:
        pick_model = st.selectbox("Model", options=list(selected_models.keys()))
    with c3:
        _, mdl_labels, _ = selected_models[pick_model]
        pick_label = st.selectbox("Label", options=mdl_labels)

    gen_cam = st.button("Generate Grad-CAM")

    if gen_cam:
        pack = ss["preproc_cache"][pick_img_key]
        x224 = pack["x224"]
        base_u8 = pack["viz_crop224"]
        mdl, mdl_labels, _ = selected_models[pick_model]
        try:
            idx = [l.lower() for l in mdl_labels].index(pick_label.lower())
        except ValueError:
            idx = None
            st.error("Label not in selected model.")

        if idx is not None:
            grad_model, feat_name = get_or_build_grad_model(mdl)
            cam01 = grad_cam_from_feat(grad_model, x224, idx)
            overlay = overlay_heatmap(cam01, base_u8)

            colA, colB, colC = st.columns(3)
            with colA:
                st.image(base_u8, caption="Preprocessed (224×224)", width=260, clamp=True)
            with colB:
                st.image(cam01, caption="Grad-CAM (0–1)", width=260, clamp=True)
            with colC:
                st.image(overlay, caption="Overlay", width=260, clamp=True)
else:
    st.caption("Upload, preprocess, and select models to enable Grad-CAM.")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    'Created by **Alejandro Ruiz Martínez** · '
    '[Website](https://alejan-rumart.github.io/)',
)
