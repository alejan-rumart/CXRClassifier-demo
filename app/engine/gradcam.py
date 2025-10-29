# app/engine/gradcam.py
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import cv2
import tensorflow as tf
import keras

_GRADMODELS: Dict[int, Tuple[tf.keras.Model, str]] = {}

def _find_pre_gap_tensor(model: keras.Model):
    gap_layer = None
    for lyr in reversed(model.layers):
        if isinstance(lyr, tf.keras.layers.GlobalAveragePooling2D):
            gap_layer = lyr
            break
    if gap_layer is not None:
        feat = gap_layer.input
        src = getattr(feat, "_keras_history", None)
        lname = src[0].name if (src and hasattr(src[0], "name")) else "pre_gap"
        return feat, lname
    for lyr in reversed(model.layers):
        try:
            shp = lyr.output_shape
            if isinstance(shp, tuple) and len(shp) == 4:
                return lyr.output, lyr.name
        except Exception:
            continue
    raise RuntimeError("No 4D feature map found before pooling.")

def get_or_build_grad_model(model: keras.Model) -> Tuple[tf.keras.Model, str]:
    key = id(model)
    if key in _GRADMODELS:
        return _GRADMODELS[key]
    feat_tensor, feat_name = _find_pre_gap_tensor(model)
    grad_model = tf.keras.Model(model.input, [feat_tensor, model.output])
    dummy = tf.zeros((1,) + model.input_shape[1:], dtype=tf.float32)
    _ = grad_model(dummy, training=False)
    _GRADMODELS[key] = (grad_model, feat_name)
    return grad_model, feat_name

def grad_cam_from_feat(grad_model: tf.keras.Model, x_m1p1_224: np.ndarray, class_index: int) -> np.ndarray:
    h, w = x_m1p1_224.shape[:2]
    x = x_m1p1_224[None, ...].astype(np.float32)
    with tf.GradientTape() as tape:
        feat, logits = grad_model(x, training=False)
        yc = logits[:, class_index]
    grads = tape.gradient(yc, feat)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * feat, axis=-1))[0].numpy()
    cam = np.nan_to_num(cam).astype(np.float32)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam[:] = 0.0
    return cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

def overlay_heatmap(cam01: np.ndarray, gray_u8: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    base = gray_u8
    if base.ndim == 3 and base.shape[-1] == 1:
        base = base[..., 0]
    base_rgb = np.stack([base]*3, axis=-1)
    heat = cv2.applyColorMap((cam01 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1.0 - alpha) * base_rgb).clip(0, 255).astype(np.uint8)
