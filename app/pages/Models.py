# app/pages/Models.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Models", page_icon="🧪", layout="wide")

APP_DIR = Path(__file__).resolve().parents[1]   # .../app
MODELS_DIR = APP_DIR / "models"

# Hide Streamlit toolbar/menu/footer on this page too
st.markdown("""
<style>
[data-testid="stToolbar"] {display: none !important;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none !important;}
</style>
""", unsafe_allow_html=True)

def _discover_model_dirs() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.iterdir()):
            if p.is_dir():
                out[p.name] = p
    return out

def _find_eval_report(model_dir: Path) -> Optional[Path]:
    # Accept both names
    for name in ("evaluation_report.json", "eval_report.json"):
        p = model_dir / name
        if p.exists():
            return p
    return None

def _load_json_lenient(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8", errors="replace")
    # Remove UTF-8 BOM if present
    if txt and txt[0] == "\ufeff":
        txt = txt.lstrip("\ufeff")
    # Replace NaN/Infinity with null (JSON doesn't allow them)
    txt = re.sub(r'\bNaN\b', 'null', txt)
    txt = re.sub(r'\b-?Infinity\b', 'null', txt)
    # Escape stray backslashes not forming a valid JSON escape
    # Valid escapes: \\, \/, \", \b, \f, \n, \r, \t, \uXXXX
    txt = re.sub(r'\\(?![\\/"bfnrtu])', r'\\\\', txt)
    return json.loads(txt)

def _safe_get(d: dict, path: List[str], default=None):
    x = d
    for k in path:
        if not isinstance(x, dict) or k not in x:
            return default
        x = x[k]
    return x

def _parse_report(report_path: Path) -> Tuple[List[str], Dict[str, float], Optional[float]]:
    rpt = _load_json_lenient(report_path)

    # Label order (prefer explicit list)
    label_names: List[str] = (
        rpt.get("label_names")
        or rpt.get("target_labels")
        or list((rpt.get("per_label") or {}).keys())
        or []
    )

    per_label_auc: Dict[str, float] = {}
    per_label = rpt.get("per_label") or {}

    for lbl in per_label.keys():
        # Try common locations for AUCs
        cand = (
            _safe_get(per_label[lbl], ["test_metrics", "roc_auc"]) or
            _safe_get(per_label[lbl], ["roc_auc_test"]) or
            _safe_get(per_label[lbl], ["val_metrics", "roc_auc"]) or
            _safe_get(per_label[lbl], ["validation_metrics", "roc_auc"]) or
            per_label.get(lbl)  # sometimes raw number
        )
        if cand is not None:
            try:
                per_label_auc[lbl] = float(cand)
            except Exception:
                pass

    # Macro: try several shapes
    macro = (
        _safe_get(rpt, ["macro_test_metrics", "roc_auc"]) or
        _safe_get(rpt, ["macro", "test", "roc_auc"]) or
        _safe_get(rpt, ["macro", "roc_auc"]) or
        rpt.get("roc_auc_macro") or
        None
    )

    # If macro missing, compute mean of per-label if available
    if macro is None and per_label_auc:
        macro = float(np.mean([v for v in per_label_auc.values() if v is not None]))

    macro_auc: Optional[float] = float(macro) if macro is not None else None
    return label_names, per_label_auc, macro_auc

@st.cache_data(show_spinner=False)
def _build_auc_table() -> Tuple[pd.DataFrame, List[str]]:
    discovered = _discover_model_dirs()
    if not discovered:
        return pd.DataFrame(), []

    labels_union: List[str] = []
    per_model: Dict[str, Tuple[List[str], Dict[str, float], Optional[float]]] = {}
    failed: List[str] = []

    for model_name, model_dir in discovered.items():
        rpt_path = _find_eval_report(model_dir)
        if rpt_path is None:
            failed.append(f"{model_name}: no evaluation_report.json")
            continue
        try:
            labels, per_auc, macro = _parse_report(rpt_path)
        except Exception as e:
            failed.append(f"{model_name}: parse error — {e}")
            continue

        per_model[model_name] = (labels, per_auc, macro)
        for lbl in labels:
            if lbl not in labels_union:
                labels_union.append(lbl)

    if not per_model:
        return pd.DataFrame(), failed

    row_index = labels_union + ["Macro"]
    df = pd.DataFrame(index=row_index)

    for model_name, (labels, per_auc, macro) in per_model.items():
        col_vals = [per_auc.get(lbl) for lbl in labels_union]
        col_vals.append(macro)
        df[model_name] = col_vals

    return df, failed

def main():
    st.title("Models")
    st.write(
        "Summary of per-label ROC-AUC (on TEST set of PadChest database) for each available model. "
        "For a proper interpretation of these numbers, please read the thesis."
    )

    df, failed = _build_auc_table()
    if failed:
        with st.expander("Reports with issues", expanded=False):
            for item in failed:
                st.write(f"• {item}")

    if df.empty:
        st.warning(
            "No evaluation reports parsed successfully. Expected one of:\n"
            "`app/models/<model>/evaluation_report.json` or `app/models/<model>/eval_report.json`."
        )
        return

    st.dataframe(
        df.round(3),
        use_container_width=True,
        height=min(600, 50 + 30 * len(df)),
    )

if __name__ == "__main__":
    main()
