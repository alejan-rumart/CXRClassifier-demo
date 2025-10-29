# app/engine/dicom_io.py
from __future__ import annotations
from typing import Tuple, BinaryIO
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# NEW: explicit exception to signal unsupported compression / missing codecs
class UnsupportedCompressedDicomError(RuntimeError):
    pass

def is_dicom(filelike) -> bool:
    """Conservatively detect DICOM.
    - Return True immediately if 'DICM' preamble is present.
    - Otherwise try a forced header parse, but only accept it if key DICOM tags exist.
    - Avoid misclassifying common raster formats by checking their magics first.
    """
    pos = filelike.tell()
    try:
        filelike.seek(0)
        head = filelike.read(132)

        # Fast reject: common raster signatures
        # PNG
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return False
        # JPEG/JFIF/EXIF
        if head.startswith(b"\xFF\xD8\xFF"):
            return False
        # TIFF (II*/MM*)
        if head.startswith(b"II*\x00") or head.startswith(b"MM\x00*"):
            return False
        # BMP
        if head.startswith(b"BM"):
            return False

        # Classic Part-10 check
        if len(head) >= 132 and head[128:132] == b"DICM":
            return True

        # Fallback: try to parse minimal header as DICOM
        filelike.seek(0)
        try:
            ds = pydicom.dcmread(filelike, stop_before_pixels=True, force=True)
        except Exception:
            return False

        # Accept only if it "looks like" an image DICOM:
        # - Have at least one of canonical identifiers AND
        # - Either PixelData exists, or Rows+Columns are present
        has_uid = hasattr(ds, "SOPClassUID") or hasattr(ds, "StudyInstanceUID") or hasattr(ds, "SeriesInstanceUID")
        has_image_shape = ("PixelData" in ds) or (hasattr(ds, "Rows") and hasattr(ds, "Columns"))
        return bool(has_uid and has_image_shape)

    finally:
        filelike.seek(pos)



def _to_uint8_windowed(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Apply VOI LUT / windowing and map to uint8; handle MONOCHROME1 inversion."""
    # Apply VOI LUT if present (returns int16/float), else use raw
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    # Convert to float and window to [0,1]
    arr = arr.astype(np.float32)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    # Invert MONOCHROME1 (white=low)
    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        arr = 1.0 - arr

    # Map to uint8
    return np.clip((arr * 255.0).round(), 0, 255).astype(np.uint8)


def load_dicom(filelike: BinaryIO) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return (preview_u8, raw_pixels, metadata). Raises UnsupportedCompressedDicomError when
    pixel data cannot be decoded due to missing codecs (e.g., JPEG-LS, JPEG2000)."""
    pos = filelike.tell()
    try:
        filelike.seek(0)
        ds = pydicom.dcmread(filelike, force=True)

        # Decode pixel data
        try:
            arr = ds.pixel_array  # may raise if required decoders aren't installed
        except Exception as e:
            tsuid = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", ""))
            raise UnsupportedCompressedDicomError(
                f"Cannot decode DICOM pixel data (TransferSyntaxUID={tsuid}). "
                "Install DICOM codecs (e.g., 'pylibjpeg[all]' or 'gdcm') or export as PNG/JPG."
            ) from e

        # Some modalities can be multi-frame; reduce to first frame for preview if needed
        if arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr[0]

        preview_u8 = _to_uint8_windowed(arr, ds)

        meta = {
            "PhotometricInterpretation": str(getattr(ds, "PhotometricInterpretation", "")),
            "TransferSyntaxUID": str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "")),
            "Rows": int(getattr(ds, "Rows", preview_u8.shape[0])),
            "Columns": int(getattr(ds, "Columns", preview_u8.shape[1])),
            "BitsStored": int(getattr(ds, "BitsStored", 0)),
            "BitsAllocated": int(getattr(ds, "BitsAllocated", 0)),
            "Modality": str(getattr(ds, "Modality", "")),
            "ViewPosition": str(getattr(ds, "ViewPosition", "")),
            "BodyPartExamined": str(getattr(ds, "BodyPartExamined", "")),
        }
        return preview_u8, arr, meta
    finally:
        filelike.seek(pos)
