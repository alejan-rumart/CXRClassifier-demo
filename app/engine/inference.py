# app/engine/inference.py
from pathlib import Path
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import json
from typing import Dict

def load_keras_model(model_path: Path):
    return keras.models.load_model(str(model_path), compile=False)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict_logits(model, x224: np.ndarray) -> np.ndarray:
    x = np.expand_dims(x224.astype(np.float32), axis=0)
    preds = model.predict(x, verbose=0)
    return np.asarray(preds).reshape(-1).astype(np.float32)

def predict_probs(model, x224: np.ndarray) -> np.ndarray:
    logits = predict_logits(model, x224)
    return _sigmoid(logits)

_ARCH_SHORT = {"efficientnet_b0": "EN-B0", "efficientnetb0": "EN-B0"}

def _pretty_name_from_runconfig(rc_path: Path) -> str:
    try:
        cfg = json.loads(rc_path.read_text(encoding="utf-8"))
    except Exception:
        return rc_path.parent.name
    arch = str(cfg.get("architecture", "")).lower()
    arch_short = _ARCH_SHORT.get(arch, arch.upper() or "MODEL")
    cutmix = "CutMix" if bool(cfg.get("use_cutmix", False)) else "NoMix"
    return f"{arch_short} · {cutmix} · best"

def discover_model_dirs(models_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not models_root.exists():
        return out
    for p in sorted(models_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "model.keras").exists() and (p / "run_config.json").exists():
            disp = _pretty_name_from_runconfig(p / "run_config.json")
            name = disp if disp not in out else f"{disp} ({p.name})"
            out[name] = p
    return out

def load_model_labels_from_dir(model_dir: Path) -> list[str]:
    rc = Path(model_dir) / "run_config.json"
    try:
        cfg = json.loads(rc.read_text(encoding="utf-8"))
        if "target_labels" in cfg and cfg["target_labels"]:
            return [str(s).strip().lower() for s in cfg["target_labels"]]
        if "selected_label_idx" in cfg and "all_label_names" in cfg:
            idx = cfg["selected_label_idx"]
            alls = cfg["all_label_names"]
            return [str(alls[i]).strip().lower() for i in idx]
    except Exception:
        pass
    return []

def load_model_operating_points_from_dir(model_dir: Path) -> dict | None:
    csvp = Path(model_dir) / "operating_points_table.csv"
    if not csvp.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(csvp)
        df = df[df["label"].str.lower() != "macro"].copy()
        cols = {c.lower(): c for c in df.columns}
        lab = cols.get("label") or list(df.columns)[0]
        sens_col = next((v for k, v in cols.items() if ("sens" in k and "thr" in k)), None)
        spec_col = next((v for k, v in cols.items() if (("spec" in k or "esp" in k) and "thr" in k)), None)
        if lab and sens_col and spec_col:
            sens = dict(zip(df[lab].astype(str).str.lower(), df[sens_col].astype(float)))
            spec = dict(zip(df[lab].astype(str).str.lower(), df[spec_col].astype(float)))
            return {"sens": sens, "spec": spec}
    except Exception:
        return None
    return None

__all__ = [
    "load_keras_model", "predict_logits", "predict_probs",
    "discover_model_dirs", "load_model_labels_from_dir",
    "load_model_operating_points_from_dir",
]
