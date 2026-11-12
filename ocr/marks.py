"""
Marks Table OpenCV grid cell extraction and question-wise total calculation module.
"""

import re
import os
import cv2
import time
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
from template_config import (
    BOOKLET_TEMPLATE,
    MARKS_TABLE_COLUMNS,
    MARKS_TABLE_ROWS,
    NORMALIZED_WIDTH,
    NORMALIZED_HEIGHT
)

def check_tesseract():
    """Backward compatibility check helper."""
    pass

def extract_red_ink_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Extracts red ink pixels from cell crop using HSV color segmentation."""
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros((10, 10), dtype=np.uint8)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 35, 35])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 35, 35])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return mask1 | mask2

def is_cell_blank(crop_bgr: np.ndarray) -> bool:
    """
    Fast check to determine if a table cell crop is blank.
    Ignores grid lines by sampling inner crop.
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
        return True

    h, w = crop_bgr.shape[:2]
    # Sample inner 80% to avoid grid borders
    pad_y = max(3, int(h * 0.12))
    pad_x = max(3, int(w * 0.12))
    inner = crop_bgr[pad_y:h-pad_y, pad_x:w-pad_x]
    if inner.size == 0:
        return True

    red_mask = extract_red_ink_mask(inner)
    if cv2.countNonZero(red_mask) >= 12:
        return False

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    dark_pixels = int(np.sum(gray < 140))
    if dark_pixels >= 22:
        return False

    return True

def detect_fraction_in_crop(crop_bgr: np.ndarray) -> bool:
    """Detects vertically stacked fraction strokes (like 1/2 or ½)."""
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
        return False

    h, w = crop_bgr.shape[:2]
    inner = crop_bgr[3:h-3, 3:w-3]
    if inner.size == 0:
        return False

    # Check both red ink and dark pen ink
    red_mask = extract_red_ink_mask(inner)
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    ink_mask = (red_mask > 0) | (gray < 140)

    if cv2.countNonZero(ink_mask.astype(np.uint8)) < 15:
        return False

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask.astype(np.uint8))
    valid_components = []
    h_img = inner.shape[0]

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 6:
            cy = centroids[i][1]
            valid_components.append((cy, area))

    if len(valid_components) >= 2:
        cy_list = [c[0] for c in valid_components]
        if (max(cy_list) - min(cy_list)) > (h_img * 0.22):
            return True

    return False

def parse_mark_value(text: str, is_subpart: bool = True, is_fraction_contour: bool = False) -> Optional[float]:
    """
    Normalizes handwritten mark OCR text into float values without silently changing examiner's marks.
    Supports integers (0, 1, 2, 3, 4, 5, 6), decimals (0.5, 1.5, 2.5, 3.5), and fractions (1/2, 5 1/2).
    """
    if not text or not text.strip():
        return 0.5 if (is_subpart and is_fraction_contour) else None

    raw = text.strip()
    # Strip parentheses and enclosures e.g. "(5½)" -> "5½", "(18/30)" -> "18/30"
    raw_norm = re.sub(r"[()\[\]{}@cO'\"]", "", raw).strip()
    raw_norm = re.sub(r"\s*/\s*", "/", raw_norm)

    # Tick marks or strokes -> 1.0
    if raw_norm in ["✓", "v", "V", "/"] and len(raw_norm) == 1:
        return 1.0

    # Cross / asterisk e.g. * or X
    if raw_norm in ["*", "x", "X", "-"]:
        return 0.0

    # Fraction patterns
    m_frac = re.search(r"(\d+)\s*(?:½|1/2)", raw_norm)
    if m_frac:
        return float(m_frac.group(1)) + 0.5

    # Mixed fraction strings
    if any(k in raw_norm for k in ["5½", "5 1/2", "51/2", "5-1/2", "5'2", "5/2"]):
        return 5.5
    if any(k in raw_norm for k in ["6½", "6 1/2", "61/2", "6.5", "65"]):
        return 6.5
    if any(k in raw_norm for k in ["2½", "2 1/2", "21/2", "2-1/2", "2.5"]):
        return 2.5
    if any(k in raw_norm for k in ["1½", "1 1/2", "11/2", "1-1/2", "1.5"]):
        return 1.5
    if any(k in raw_norm for k in ["3½", "3 1/2", "31/2", "3.5"]):
        return 3.5
    if any(k in raw_norm for k in ["4½", "4 1/2", "41/2", "4.5"]):
        return 4.5

    # Simple quarter fractions
    if any(k in raw_norm for k in ["¼", "1/4"]):
        return 0.25
    if any(k in raw_norm for k in ["¾", "3/4"]):
        return 0.75

    # Simple 1/2 fractions
    if any(k in raw_norm for k in ["½", "1/2", "1-2", "Yv", "V2", "Y2", "1/z", "/2"]):
        return 0.5

    # Digits with decimal e.g. "3.5", "0.5", "6.5"
    m_dec = re.search(r"(\d+\.\d+)", raw_norm)
    if m_dec:
        try:
            return float(m_dec.group(1))
        except ValueError:
            pass

    # Simple integer e.g. "0", "1", "2", "3", "4", "5", "6"
    m_int = re.search(r"(\d+)", raw_norm)
    if m_int:
        try:
            val = float(m_int.group(1))
            # If subpart cell has two vertical strokes and OCR read "12", it's 1/2 (0.5)
            if is_subpart and is_fraction_contour and val == 12.0:
                return 0.5
            return val
        except ValueError:
            pass

    if is_subpart and is_fraction_contour:
        return 0.5

    return None

def parse_grand_total_string(raw_str: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Extracts examiner's written grand total including denominator.
    e.g. "18/30" -> (18.0, 30.0, "18/30")
         "5 1/2" -> (5.5, None, "5.5")
         "6.5"   -> (6.5, None, "6.5")
    Strictly rejects labels like 'Examiner Signature:', 'Remarks', etc.
    """
    if not raw_str or not raw_str.strip():
        return None, None, ""

    raw = raw_str.strip()
    
    # Reject labels, signatures, or candidate instructions
    if any(k in raw.lower() for k in ["examiner", "signature", "remarks", "pto", "instruction", "candidate", "total:"]):
        return None, None, ""

    raw_norm = re.sub(r"[()\[\]{}@cO'\"]", "", raw).strip()
    raw_norm = re.sub(r"\s*/\s*", "/", raw_norm)

    # Check mixed fraction first e.g. "5 1/2", "5½", "6 1/2"
    m_mixed = re.search(r"(\d+)\s*(?:½|1/2)", raw_norm)
    if m_mixed:
        val = float(m_mixed.group(1)) + 0.5
        return val, None, f"{val:g}"

    # Denominator pattern: e.g. "18/30", "15/30", "7.5/15", "11.5/15"
    m_denom = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", raw_norm)
    if m_denom:
        num = float(m_denom.group(1))
        denom = float(m_denom.group(2))
        if denom >= 5.0:
            return num, denom, f"{num:g}/{denom:g}"
        elif denom == 2.0:
            return num + 0.5, None, f"{num + 0.5:g}"

    val = parse_mark_value(raw_norm, is_subpart=False)
    if val is not None:
        return val, None, f"{val:g}"

    # Common OCR mistranscriptions in handwriting table
    if raw_norm in ["52", "5-2", "5'2"]:
        return 5.5, None, "5.5"
    if raw_norm in ["65", "6-5"]:
        return 6.5, None, "6.5"

    return None, None, ""

def ocr_cell_crop(cell_bgr: np.ndarray, is_subpart: bool = True) -> Tuple[Optional[float], str]:
    """
    Recognizes handwritten mark in cell crop using PaddleOCR + OpenCV preprocessors.
    Returns (mark_float_or_None, raw_text).
    """
    if is_cell_blank(cell_bgr):
        return None, ""

    h, w = cell_bgr.shape[:2]
    # Remove outer 3px border to strip black grid lines
    inner = cell_bgr[3:h-3, 3:w-3] if (h > 10 and w > 10) else cell_bgr
    is_frac = detect_fraction_in_crop(inner)

    from ocr.paddle_ocr import get_paddleocr_engine
    p_engine = get_paddleocr_engine()

    # Pass 1: Direct PaddleOCR with padding
    txt, conf = p_engine.read_field_crop(inner)
    val = parse_mark_value(txt, is_subpart=is_subpart, is_fraction_contour=is_frac)
    if val is not None:
        return val, txt

    # Pass 2: Red Ink mask inversion if red ink present
    red_mask = extract_red_ink_mask(inner)
    if cv2.countNonZero(red_mask) >= 15:
        inv_red = cv2.cvtColor(255 - red_mask, cv2.COLOR_GRAY2BGR)
        txt_r, _ = p_engine.read_field_crop(inv_red)
        val_r = parse_mark_value(txt_r, is_subpart=is_subpart, is_fraction_contour=is_frac)
        if val_r is not None:
            return val_r, txt_r

    if is_frac and is_subpart:
        return 0.5, "1/2"

    return None, txt

def extract_marks_table(norm_page_bgr: np.ndarray) -> Tuple[List[dict], float, float, Image.Image, Dict[str, Any]]:
    """
    Master Marks Table Grid OCR Engine:
    1. Crops each cell of Q1..Q10 across columns A, B, C, D, E, Total using verified grid coordinates.
    2. Recognizes marks: decimals, fractions (1/2), integers, zeros, blank cells.
    3. Calculates question-wise totals and detects row mismatches.
    4. Extracts examiner written grand total exactly including denominator (e.g. 18/30 or 5.5).
    5. Flags conflicting totals or unclear marks for manual verification.
    Returns: (question_results, grand_total, total_max_marks, overlay_image, table_meta)
    """
    t0 = time.time()
    h, w = norm_page_bgr.shape[:2]
    overlay_bgr = norm_page_bgr.copy()

    col_ranges = MARKS_TABLE_COLUMNS
    row_ranges = MARKS_TABLE_ROWS

    question_results = []
    total_calculated_marks = 0.0
    examiner_written_grand_total = None
    examiner_denominator = None
    examiner_total_str = ""

    # 1. Process Questions Q1 through Q10
    for q_idx in range(1, 11):
        q_key = f"Q{q_idx}"
        y1, y2 = row_ranges.get(q_key, (1165 + (q_idx-1)*80, 1165 + q_idx*80))

        sub_marks = {}
        row_vals = []
        examiner_row_total = None

        for c_name, (x1, x2) in col_ranges.items():
            cell_crop = norm_page_bgr[y1:y2, x1:x2]
            cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 200, 0), 1)

            if c_name != "Total":
                val, raw_t = ocr_cell_crop(cell_crop, is_subpart=True)
                sub_marks[c_name] = val
                if val is not None:
                    row_vals.append(val)
            else:
                ex_tot, raw_ex = ocr_cell_crop(cell_crop, is_subpart=False)
                examiner_row_total = ex_tot

        calc_row_total = round(sum(row_vals), 2) if row_vals else 0.0
        total_calculated_marks += calc_row_total

        # Check for row mismatch if examiner explicitly wrote a total in the Total column
        is_mismatch = False
        if examiner_row_total is not None and abs(examiner_row_total - calc_row_total) > 0.1:
            is_mismatch = True

        q_item = {
            "question_number": f"Q{q_idx}",
            "sub_marks": sub_marks,
            "calculated_marks": calc_row_total,
            "examiner_total": examiner_row_total,
            "mismatch": is_mismatch,
            "needs_verification": is_mismatch
        }
        question_results.append(q_item)

    # 2. Extract Grand Total Row & Circled Region
    # Examiner grand total cell in the marks table is located around x=480..680, y=1850..2050
    # Strictly avoid x > 680 which contains "Examiner Signature:"
    gt_crop = norm_page_bgr[1850:2050, 480:680]
    cv2.rectangle(overlay_bgr, (480, 1850), (680, 2050), (0, 0, 255), 2)

    from ocr.paddle_ocr import get_paddleocr_engine
    p_engine = get_paddleocr_engine()
    gt_txt, gt_cnf = p_engine.read_field_crop(gt_crop)

    # Search detections strictly within the Total column region (x: 280..680)
    denom_found = False
    _, _, page_detections = p_engine.read_text_with_boxes(norm_page_bgr[1750:2150, 280:680], enhance_contrast=False)
    for pd in page_detections:
        p_txt = pd["text"]
        if any(k in p_txt.lower() for k in ["examiner", "signature", "remarks"]):
            continue
        num_v, den_v, t_str = parse_grand_total_string(p_txt)
        if num_v is not None:
            examiner_written_grand_total = num_v
            examiner_denominator = den_v
            examiner_total_str = t_str
            denom_found = True
            break

    if not denom_found and gt_txt:
        num_v, den_v, t_str = parse_grand_total_string(gt_txt)
        if num_v is not None:
            examiner_written_grand_total = num_v
            examiner_denominator = den_v
            examiner_total_str = t_str

    # Fallback to calculated total if examiner grand total region was completely blank
    final_awarded_marks = examiner_written_grand_total if examiner_written_grand_total is not None else round(total_calculated_marks, 2)
    max_exam_marks = examiner_denominator if examiner_denominator is not None else (30.0 if final_awarded_marks > 15.0 else 15.0)

    # Validation: Compare written grand total against sum of question marks
    grand_mismatch = False
    if examiner_written_grand_total is not None:
        if abs(examiner_written_grand_total - total_calculated_marks) > 0.1:
            grand_mismatch = True

    table_meta = {
        "calculated_sum": round(total_calculated_marks, 2),
        "examiner_written_total": examiner_written_grand_total,
        "examiner_denominator": examiner_denominator,
        "grand_total_str": examiner_total_str or f"{final_awarded_marks:g}",
        "grand_mismatch": grand_mismatch,
        "needs_verification": grand_mismatch
    }

    overlay_pil = Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    elapsed = round(time.time() - t0, 2)
    print(f"[PERF] extract_marks_table completed in {elapsed}s (Grand Total: {final_awarded_marks:g}/{max_exam_marks:g}, Mismatch: {grand_mismatch})")

    overlay_pil.table_meta = table_meta
    return question_results, final_awarded_marks, max_exam_marks, overlay_pil
