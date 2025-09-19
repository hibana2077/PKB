"""Patch-level operations for PKB.

These functions are defined in a separate module so they are picklable
when used inside DataLoader worker processes on Windows.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as F


def patch_op_highpass(pil_img: Image.Image) -> Image.Image:
    """Approximate high-pass enhancement via UnsharpMask (strong sharpening)."""
    return pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))


def patch_op_paper_noise(pil_img: Image.Image) -> Image.Image:
    """Add mild Gaussian paper-like noise to the patch."""
    arr = np.array(pil_img).astype(np.float32)
    # noise std ~ 12/255; tweakable if needed
    noise = np.random.normal(0.0, 12.0, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return F.to_pil_image(arr)


def patch_op_color_jitter(pil_img: Image.Image) -> Image.Image:
    """Apply color jitter to the patch (light)."""
    cj = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    return cj(pil_img)


def resolve_pkb_patch_op(name: str) -> Optional[Callable[[Image.Image], Image.Image]]:
    """Map CLI name to callable or None for default blur."""
    if name == 'blur':
        return None  # use default Gaussian blur with sigma inside PKB
    if name == 'highpass':
        return patch_op_highpass
    if name == 'paper-noise':
        return patch_op_paper_noise
    if name == 'color-jitter':
        return patch_op_color_jitter
    raise ValueError(f"Unknown pkb patch op: {name}")


__all__ = [
    'patch_op_highpass',
    'patch_op_paper_noise',
    'patch_op_color_jitter',
    'resolve_pkb_patch_op',
]
