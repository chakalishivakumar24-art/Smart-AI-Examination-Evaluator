"""
Template configuration & ROI box visualization module for ICFAI answer booklets.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple
from template_config import BOOKLET_TEMPLATE, NORMALIZED_WIDTH, NORMALIZED_HEIGHT

def normalize_to_canvas(img_bgr: np.ndarray, target_w: int = NORMALIZED_WIDTH, target_h: int = NORMALIZED_HEIGHT) -> np.ndarray:
    """Normalizes image array to target canvas resolution (1600 x 2200)."""
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    h, w = img_bgr.shape[:2]
    if w == target_w and h == target_h:
        return img_bgr
    return cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

def generate_roi_overlay_image(norm_bgr: np.ndarray, rois: Dict[str, Tuple[int, int, int, int]] = None) -> Image.Image:
    """Draws color-coded ROI boxes and field labels on the normalized 1600 x 2200 Page 1 canvas."""
    if rois is None:
        rois = BOOKLET_TEMPLATE

    overlay = norm_bgr.copy()
    for field_name, (x, y, w, h) in rois.items():
        color = (0, 255, 0) if "signature" not in field_name else (255, 165, 0)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(overlay, field_name, (x + 5, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
