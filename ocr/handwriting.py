"""
Handwritten ROI extraction & multi-pass OCR candidate comparison engine.
"""

import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Tuple, List
from utils.text_parser import clean_ocr_value

_TESSERACT_AVAILABLE = True
try:
    tess_paths = [
        os.getenv("TESSERACT_CMD"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
    ]
    for p in tess_paths:
        if p and os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break
except Exception:
    _TESSERACT_AVAILABLE = False

def check_tesseract():
    if not _TESSERACT_AVAILABLE:
        raise RuntimeError("Tesseract OCR is not installed. Please install Tesseract OCR.")

def extract_field_crop_multipass(
    crop_bgr: np.ndarray,
    field_name: str,
    psm_config: str = "--psm 6"
) -> Tuple[str, float, Image.Image, str, int]:
    """
    Evaluates multi-pass tailored preprocessing variants for field crop OCR:
    1. Grayscale + 15px White Padding to eliminate boundary truncation.
    2. 2.5x Cubic Upscaling for fine handwritten strokes.
    3. Contrast enhancement (CLAHE), Adaptive Gaussian Thresholding, and Otsu Binarization.
    4. Evaluates OCR candidates via pytesseract.Output.DICT to select highest confidence valid text.
    """
    check_tesseract()
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
        empty_pil = Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8))
        return "", 0.0, empty_pil, "none", 0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    
    # 15px white padding
    padded = cv2.copyMakeBorder(gray, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)

    # 2.5x Cubic Upscale
    gray_upscaled = cv2.resize(padded, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    # Noise Reduction & Contrast Enhancement
    denoised = cv2.fastNlMeansDenoising(gray_upscaled, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe_img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(denoised)
    adaptive_img = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    _, otsu_img = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    variants = [
        ("padded_raw", gray_upscaled),
        ("clahe", clahe_img),
        ("adaptive", adaptive_img),
        ("otsu", otsu_img)
    ]

    pass_results = []

    for idx, (v_name, v_img) in enumerate(variants, start=1):
        v_pil = Image.fromarray(v_img)
        try:
            c_dict = pytesseract.image_to_data(v_pil, config=psm_config, output_type=pytesseract.Output.DICT)
            raw_words = []
            confs = []
            for t, c in zip(c_dict.get('text', []), c_dict.get('conf', [])):
                st_t = str(t).strip()
                if st_t:
                    raw_words.append(st_t)
                    try:
                        c_val = float(c)
                        if c_val > 0:
                            confs.append(c_val)
                    except (ValueError, TypeError):
                        pass

            if not raw_words:
                continue

            raw_txt = " ".join(raw_words).strip()
            avg_conf = float(np.mean(confs)) if confs else 0.0

            cleaned = clean_ocr_value(raw_txt, field_name)
            if cleaned:
                pass_results.append((cleaned, avg_conf, v_pil, v_name, idx, raw_txt))
                if avg_conf >= 70.0:
                    break
        except Exception:
            continue

    if not pass_results:
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        return "", 0.0, crop_pil, "none", 0

    pass_results.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    best_txt, best_conf, best_pil, best_var, best_pass, _ = pass_results[0]

    return best_txt, round(best_conf, 1), best_pil, best_var, best_pass
