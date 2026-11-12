"""
Orientation detection and automatic rotation module for ICFAI Examination Booklets.
Supports robust orthogonal page angle detection (0, 90, 180, 270 degrees) using:
- PaddleOCR deep learning text recognition and keyword matching (primary)
- Document layout analysis (table grid lines in lower half vs upper half)
- Tesseract OCR keyword fallback (if available)
- Coordinate transformations for detection boxes and template ROIs
"""

import os
import cv2
import time
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional, Any

# Primary Header Keywords for ICFAI / University Examination Master Booklet Page 1
HEADER_KEYWORDS = [
    "icfai", "tech", "faculty", "science", "technology", "booklet", "serial",
    "comprehensive", "examination", "student", "name", "enrollment", "course",
    "title", "code", "semester", "branch", "section", "date", "invigilator",
    "instructor", "maximum", "marks", "university", "higher", "education",
    "hyderabad", "signature", "instructions", "class", "test", "regular", "program"
]

# Common Answer Sheet Keywords for subsequent pages (Pages 2+)
ANSWER_KEYWORDS = [
    "question", "answer", "ans", "solution", "section", "part",
    "step", "page", "total", "marks", "hence", "therefore", "let"
]

# Check Tesseract availability as an auxiliary fallback
_TESSERACT_AVAILABLE = False
try:
    import pytesseract
    tess_paths = [
        os.getenv("TESSERACT_CMD"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
    ]
    for p in tess_paths:
        if p and os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            _TESSERACT_AVAILABLE = True
            break
except Exception:
    _TESSERACT_AVAILABLE = False


def rotate_image_angle(image_bgr: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotates an OpenCV BGR image by 0, 90, 180, or 270 degrees clockwise.
    Angle represents clockwise rotation applied to bring image to upright portrait:
    - 0: No rotation
    - 90: Rotate 90° clockwise
    - 180: Rotate 180°
    - 270: Rotate 270° clockwise (90° counterclockwise)
    """
    if image_bgr is None or image_bgr.size == 0 or angle == 0:
        return image_bgr

    if angle == 90:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image_bgr, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_bgr


def rotate_image_to_upright(image_bgr: np.ndarray, angle: int) -> np.ndarray:
    """Convenience alias for rotate_image_angle."""
    return rotate_image_angle(image_bgr, angle)


def ensure_portrait_orientation(image_bgr: np.ndarray) -> np.ndarray:
    """
    Ensures image is in portrait orientation (height > width).
    If landscape, rotates 90° clockwise as a baseline check.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if w > h * 1.05:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    return image_bgr


def score_orientation_candidate(
    cand_bgr: np.ndarray,
    is_first_page: bool = True,
    sample_crop_ratio: float = 0.35
) -> float:
    """
    Scores how upright a candidate portrait image is.
    Higher score indicates higher confidence of upright orientation.
    Uses PaddleOCR text recognition on the top portion, keyword density,
    line confidences, and OpenCV layout table grid distribution.
    """
    if cand_bgr is None or cand_bgr.size == 0:
        return -100.0

    h, w = cand_bgr.shape[:2]
    # Sample top region where header lives in upright booklet
    crop_h = max(30, int(h * sample_crop_ratio))
    top_crop = cand_bgr[0:crop_h, :]

    # Scale crop to max 900px for high-speed OCR evaluation
    ch, cw = top_crop.shape[:2]
    if max(ch, cw) > 900:
        scale = 900.0 / float(max(ch, cw))
        eval_crop = cv2.resize(top_crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
    else:
        eval_crop = top_crop

    score = 0.0
    text_extracted = ""

    # 1. Primary: PaddleOCR text recognition
    try:
        from ocr.paddle_ocr import get_paddleocr_engine
        engine = get_paddleocr_engine()
        full_text, avg_conf, detections = engine.read_text_with_boxes(eval_crop, enhance_contrast=False)
        text_extracted = full_text.lower()

        keywords = HEADER_KEYWORDS if is_first_page else (HEADER_KEYWORDS + ANSWER_KEYWORDS)
        kw_matches = sum(1 for kw in keywords if kw in text_extracted)
        # Each recognized header keyword strongly indicates upright orientation
        score += kw_matches * 60.0

        # Quality text lines with high confidence (> 65%)
        good_lines = [d for d in detections if d.get("confidence", 0) >= 65.0]
        score += len(good_lines) * 6.0
        score += avg_conf * 0.4

    except Exception as e:
        # 2. Auxiliary Fallback: Tesseract OCR if PaddleOCR encounters issues
        if _TESSERACT_AVAILABLE:
            try:
                pil_temp = Image.fromarray(cv2.cvtColor(eval_crop, cv2.COLOR_BGR2RGB))
                t_text = pytesseract.image_to_string(pil_temp, config="--psm 6").lower()
                text_extracted = t_text
                keywords = HEADER_KEYWORDS if is_first_page else (HEADER_KEYWORDS + ANSWER_KEYWORDS)
                kw_matches = sum(1 for kw in keywords if kw in t_text)
                score += kw_matches * 40.0
            except Exception:
                pass

    # 3. Fast Path: If top crop alone has >= 3 strong keywords, return early with decisive score
    if score >= 180.0:
        return score

    # 4. OpenCV Layout Analysis: Examiner Marks Table Grid Distribution (Page 1)
    # In an upright ICFAI Booklet, dense table lines reside in the lower half (y > 0.45 * h).
    # In an upside-down booklet, dense table lines reside in the upper half!
    if is_first_page:
        try:
            gray = cv2.cvtColor(cand_bgr, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
            h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * 0.04)), 1))
            v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * 0.03))))
            grid_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_k) + cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_k)

            top_grid_pts = cv2.countNonZero(grid_lines[:int(h * 0.45), :])
            bot_grid_pts = cv2.countNonZero(grid_lines[int(h * 0.45):, :])

            if bot_grid_pts > top_grid_pts * 1.15:
                score += 25.0
            elif top_grid_pts > bot_grid_pts * 1.15:
                score -= 25.0
        except Exception:
            pass

    return score


def find_best_orthogonal_orientation(
    image_bgr: np.ndarray,
    is_first_page: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Detects page rotation angle (0, 90, 180, 270 degrees) and auto-rotates image to upright.
    Evaluates only plausible candidate rotations based on aspect ratio:
    - Landscape (w > h): tests 90° CW vs 270° CW
    - Portrait (h >= w): tests 0° (current) vs 180° (flipped)
    Returns: (upright_bgr_image, detected_angle_to_apply)
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr, 0

    t0 = time.time()
    h, w = image_bgr.shape[:2]

    # CASE 1: Landscape image (uploaded sideways)
    if w > h * 1.05:
        cand_90 = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        cand_270 = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        score_90 = score_orientation_candidate(cand_90, is_first_page=is_first_page)
        # Fast path if 90 CW is clearly upright
        if score_90 >= 180.0:
            best_angle = 90
            best_img = cand_90
        else:
            score_270 = score_orientation_candidate(cand_270, is_first_page=is_first_page)
            if score_90 >= score_270:
                best_angle = 90
                best_img = cand_90
            else:
                best_angle = 270
                best_img = cand_270

    # CASE 2: Portrait image (already vertical, may be upright or upside-down 180°)
    else:
        cand_0 = image_bgr
        score_0 = score_orientation_candidate(cand_0, is_first_page=is_first_page)

        # Fast path: If current orientation already has strong header keywords, preserve it!
        # This prevents unnecessary OCR processing and preserves already correctly oriented images.
        if score_0 >= 150.0:
            best_angle = 0
            best_img = cand_0
        else:
            cand_180 = cv2.rotate(image_bgr, cv2.ROTATE_180)
            score_180 = score_orientation_candidate(cand_180, is_first_page=is_first_page)

            if score_180 > score_0:
                best_angle = 180
                best_img = cand_180
            else:
                best_angle = 0
                best_img = cand_0

    elapsed = time.time() - t0
    print(f"[ORIENTATION] Detected angle: {best_angle}° in {elapsed:.2f}s (is_first_page={is_first_page})")
    return best_img, best_angle


def detect_orientation_angle(image_bgr: np.ndarray, is_first_page: bool = True) -> int:
    """
    Detects page rotation angle (0, 90, 180, 270 degrees) without copying/modifying full image.
    Returns: angle (int)
    """
    _, angle = find_best_orthogonal_orientation(image_bgr, is_first_page=is_first_page)
    return angle


def detect_and_rotate_image(pil_img: Image.Image, is_first_page: bool = True) -> Tuple[Image.Image, int]:
    """
    Auto-rotates PIL Image to upright orientation.
    Returns (rotated_pil_image, detected_angle).
    """
    if not isinstance(pil_img, Image.Image):
        return pil_img, 0

    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    upright_bgr, angle = find_best_orthogonal_orientation(img_bgr, is_first_page=is_first_page)

    upright_pil = Image.fromarray(cv2.cvtColor(upright_bgr, cv2.COLOR_BGR2RGB))
    return upright_pil, angle


def transform_bbox(
    box: Tuple[int, int, int, int],
    orig_w: int,
    orig_h: int,
    angle: int
) -> Tuple[int, int, int, int]:
    """
    Transforms bounding box (x, y, w, h) from original image coordinates
    to rotated image coordinates after rotation by `angle` (0, 90, 180, 270).
    """
    x, y, w, h = box
    if angle == 90:
        # 90° Clockwise: W' = orig_h, H' = orig_w
        return orig_h - (y + h), x, h, w
    elif angle == 180:
        # 180°: W' = orig_w, H' = orig_h
        return orig_w - (x + w), orig_h - (y + h), w, h
    elif angle == 270:
        # 270° Clockwise (90° CCW): W' = orig_h, H' = orig_w
        return y, orig_w - (x + w), h, w
    return x, y, w, h


def inverse_transform_bbox(
    box_rot: Tuple[int, int, int, int],
    orig_w: int,
    orig_h: int,
    angle: int
) -> Tuple[int, int, int, int]:
    """
    Transforms bounding box (x', y', w', h') on rotated image back to original image coordinates.
    Exact mathematical inverse of transform_bbox.
    """
    rx, ry, rw, rh = box_rot
    if angle == 90:
        return ry, orig_h - (rx + rw), rh, rw
    elif angle == 180:
        return orig_w - (rx + rw), orig_h - (ry + rh), rw, rh
    elif angle == 270:
        return orig_w - (ry + rh), rx, rh, rw
    return rx, ry, rw, rh
