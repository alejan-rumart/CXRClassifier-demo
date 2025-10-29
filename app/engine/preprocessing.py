# app/engine/preprocessing.py
import numpy as np
from PIL import Image
import cv2
import torch
import torchxrayvision as xrv
from torchxrayvision.baseline_models.chestx_det import PSPNet as ChestXDetPSPNet

TARGET_SIZE = 224
PAD_FRACTION = 0.03
DEVICE = torch.device("cpu")

_PSPNET = None
_PSP_TARGETS = None

def _get_pspnet():
    global _PSPNET, _PSP_TARGETS
    if _PSPNET is None:
        _PSPNET = ChestXDetPSPNet().to(DEVICE).eval()
        _PSP_TARGETS = _PSPNET.targets
    return _PSPNET, _PSP_TARGETS

def _robust_uint8(img, low_pct=0.5, high_pct=99.5):
    if isinstance(img, Image.Image):
        if img.mode not in ("L", "I;16", "I", "F"):
            img = img.convert("L")
        arr = np.array(img)
    else:
        arr = np.asarray(img)
    arr = arr.astype(np.float32)
    if np.all(arr == arr.flat[0]):
        return np.zeros_like(arr, dtype=np.uint8)
    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    return (arr * 255.0).round().astype(np.uint8)

def _postprocess_lung_mask(prob_map, thresh=0.40):
    m = (prob_map >= thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 3:
        keep = stats[1:, cv2.CC_STAT_AREA].argsort()[-2:] + 1
        m = np.isin(lab, keep).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return m

def _square_crop_from_mask(im_u8_512, mask_512, pad_frac=PAD_FRACTION):
    H, W = im_u8_512.shape
    ys, xs = np.where(mask_512 > 0)
    if len(xs) == 0:
        return im_u8_512
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    side = max(y1 - y0, x1 - x0)
    pad = int(round(pad_frac * side))
    side = side + 2 * pad
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, cy - side // 2)
    x0 = max(0, cx - side // 2)
    y1 = min(H, y0 + side)
    x1 = min(W, x0 + side)
    return im_u8_512[y0:y1, x0:x1]

def preprocess_png_or_jpg(pil_img: Image.Image):
    # original robust view
    viz_orig = _robust_uint8(pil_img)

    # center crop + resize to 512 for PSPNet
    arr = viz_orig[np.newaxis, ...]
    arr = xrv.datasets.XRayCenterCrop()(arr)
    arr = xrv.datasets.XRayResizer(512)(arr)
    arr512 = arr.astype(np.float32)
    if arr512.max() > 255.0:
        arr512 = (arr512 / arr512.max()) * 255.0
    viz_psp_in = np.rint(arr512[0]).clip(0, 255).astype(np.uint8)

    # lung mask
    with torch.no_grad():
        psp, targets = _get_pspnet()
        x = torch.from_numpy(arr512).unsqueeze(0).to(DEVICE)
        probs = torch.sigmoid(psp(x))[0].cpu().numpy()

    def _find_idx(name_substr):
        for i, t in enumerate(targets):
            if name_substr.lower() in t.lower():
                return i
        return None
    li = _find_idx("Left Lung")
    ri = _find_idx("Right Lung")
    lung_prob = np.maximum(probs[li], probs[ri]) if (li is not None and ri is not None) else probs.max(axis=0)

    mask_512 = _postprocess_lung_mask(lung_prob, thresh=0.40)

    ys, xs = np.where(mask_512 > 0)
    used_lung_mask = False
    if xs.size > 0 and ys.size > 0:
        bbox_w = int(xs.max() - xs.min())
        bbox_h = int(ys.max() - ys.min())
        if bbox_w >= 64 and bbox_h >= 64:
            used_lung_mask = True

    if used_lung_mask:
        crop = _square_crop_from_mask(viz_psp_in, mask_512, pad_frac=PAD_FRACTION)
    else:
        crop = viz_psp_in

    viz_crop224 = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

    x3 = np.stack([viz_crop224, viz_crop224, viz_crop224], axis=-1).astype(np.float32)
    x3 = (x3 / 127.5) - 1.0

    return x3.astype(np.float32), viz_psp_in, viz_crop224, viz_orig, used_lung_mask
