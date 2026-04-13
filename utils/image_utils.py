"""Image I/O helpers."""
from __future__ import annotations
import numpy as np
from PIL import Image
from pathlib import Path


def load_image(src) -> Image.Image:
    """Load any image source as PIL RGB Image."""
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, (str, Path)):
        return Image.open(str(src)).convert("RGB")
    if isinstance(src, np.ndarray):
        return Image.fromarray(src)
    raise TypeError(f"Unsupported image type: {type(src)}")


def to_pil(arr: np.ndarray) -> Image.Image:
    """Numpy array to PIL Image."""
    return Image.fromarray(arr)
