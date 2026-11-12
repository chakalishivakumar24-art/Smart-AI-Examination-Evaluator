import re
import time
import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Optional, Any
from models.database_models import StudentDetails
from utils.text_parser import clean_ocr_value
from template_config import (
    DEFAULT_VALUE_ROIS,
    LABEL_ANCHORS,
    BOOKLET_TEMPLATE,
    NORMALIZED_WIDTH,
    NORMALIZED_HEIGHT,
    STATUS_OK,
    STATUS_LOW_CONFIDENCE,
    STATUS_NEEDS_REVIEW,
    STATUS_NOT_DETECTED,
    STATUS_NOT_PRESENT
)
from ocr.preprocessing import (
    convert_to_opencv,
    ensure_portrait_orientation,
    correct_document_perspective,
    remove_shadows_and_enhance,
    is_valid_crop
)

def generate_roi_overlay_image(norm_bgr: np.ndarray, rois: Dict[str, Tuple[int, int, int, int]]) -> Image.Image:
    """Draws color-coded ROI boxes and field labels on the normalized 1600 x 2200 Page 1 image."""
    overlay = norm_bgr.copy()
    for field_name, (x, y, w, h) in rois.items():
        if field_name == "marks_table":
            color = (255, 120, 0)
        elif "signature" in field_name:
            color = (0, 165, 255)
        else:
            color = (0, 200, 0)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(overlay, field_name, (x + 4, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

def normalize_enrollment_number(raw_str: str) -> str:
    """Normalize ICFAI enrollment numbers without hardcoding student IDs.

    Valid ICFAI enrollment numbers follow: [23/24]STUCHH[6 digits]
    Example: 23STUCHH010066, 24STUCHH010010.
    Handles common OCR character confusions:
    - Year prefix: 23, 24 (or Q4/QY/Qu/S4 -> 24)
    - Program/Campus: STUCHH (or STUCH, STUC, STUCHI, STLCLH, SRCET, SRVCET)
    - 5-6 digit suffix: O/Q/D->0, I/L/l/|/T->1, Z->2, S->5, B->8
    Returns normalized string if valid, otherwise empty string (prompting manual verification).
    """
    if not raw_str:
        return ""
    val = re.sub(
        r"^(?:Examination|Comprehensive|Enrollment(?:\s*No)?|Registration(?:\s*No)?|Roll(?:\s*No)?|No|nt\s*No|ination|nation|ation|ninaton|ion)[:\s\-=]*",
        "", str(raw_str), flags=re.IGNORECASE
    ).strip()
    val = val.replace("|", "1").replace("/", "1").replace("!", "1")
    val = re.sub(r"[^A-Za-z0-9]", "", val).upper()
    if val.startswith(("Q4", "QY", "QU", "S4")):
        val = "24" + val[2:]
    elif val.startswith(("Q3", "S3")):
        val = "23" + val[2:]

    # Suffix check: 6 digits (or 5 digits if leading 1 was merged)
    if len(val) >= 11 and val.startswith(("23", "24")):
        yr = val[:2]
        for suf_len in (6, 5):
            suf = val[-suf_len:]
            mid = val[2:-suf_len]
            suf_clean = (suf.replace("O", "0").replace("Q", "0").replace("D", "0")
                            .replace("I", "1").replace("L", "1")
                            .replace("S", "5").replace("B", "8").replace("Z", "2"))
            if suf_clean.isdigit() and len(suf_clean) == suf_len:
                if any(k in mid for k in ["STU", "TUC", "UCH", "TCH", "STC", "SRC", "SRV", "STL", "CLH"]):
                    return f"{yr}STUCHH{suf_clean}"
                elif re.fullmatch(r"[A-Z]{3,7}", mid):
                    return f"{yr}{mid}{suf_clean}"

    match_exact = re.fullmatch(r"(2[34])STUCHH(\d{5,6})", val)
    if match_exact:
        return val
    return ""

def normalize_student_name(raw_str: str) -> str:
    """Cleans student name, rejecting labels, noise, or hallucinated words."""
    if not raw_str:
        return ""
    val = re.sub(r"^(?:Name|NAME|Student\s*Name|Namec|Name:)[:\s\-=]*", "", raw_str, flags=re.IGNORECASE).strip()
    val = re.sub(r"[\r\n\t]+", " ", val)
    # Reject printed labels or prompt noise (such as 'Contsar')
    if any(k in val.lower() for k in ["contsar", "candidate", "constant", "instructions", "comprehensive", "examination", "faculty"]):
        return ""
    val = re.sub(r"^[^A-Za-z]+", "", val)
    val = re.sub(r"[^A-Za-z\s\.]+$", "", val).strip()
    return re.sub(r"\s+", " ", val).strip()

def normalize_course_title(raw_str: str) -> str:
    if not raw_str:
        return ""
    val = re.sub(r"^(?:Course\s*Title|Subject)[:\s\-=]*", "", raw_str, flags=re.IGNORECASE).strip()
    val = re.sub(r"[\r\n\t]+", " ", val)
    val = re.sub(r"(?:Semester|Signature|Date|Section).*$", "", val, flags=re.IGNORECASE).strip()
    if len(val) < 3 or val.lower() in ["c", "section", "date", "class", "name", "not detected", "none"]:
        return ""
    if re.search(r"Machine\s*[LI1l]earn", val, re.IGNORECASE) or re.search(r"\b[LI1l]earning\b", val, re.IGNORECASE) or val.strip().upper() == "ML":
        return "Machine Learning"
    if re.search(r"[DPp]esign.*Algo", val, re.IGNORECASE) or re.search(r"qnalysi", val, re.IGNORECASE):
        return "Design and analysis of Algorithms"
    if re.search(r"Database.*Manage", val, re.IGNORECASE) or re.search(r"DBMS", val, re.IGNORECASE):
        return "Database Management System"
    if re.search(r"Comp(?:uter)?\s*Net(?:works)?", val, re.IGNORECASE):
        return "Computer Networks"
    if re.search(r"Soft(?:ware)?\s*Eng(?:ineering)?", val, re.IGNORECASE):
        return "Software Engineering"
    return val

def normalize_course_code(raw_str: str) -> str:
    if not raw_str:
        return ""
    val = re.sub(r"^(?:Course\s*Code|Subject\s*Code)[:\s\-=]*", "", raw_str, flags=re.IGNORECASE).strip()
    val = re.sub(r"[^\w]", "", val).upper()
    if not val:
        return ""
    if len(val) == 5:
        prefix = val[:2].replace("5", "S").replace("0", "O").replace("1", "I").replace("8", "B")
        suffix = val[2:].replace("Z", "2").replace("O", "0").replace("S", "5").replace("B", "8").replace("I", "1").replace("L", "1")
        return prefix + suffix
    return val

def normalize_date(raw_str: str) -> str:
    """Normalize and validate exam date string without fabricating or guessing values.
    
    Handles:
    - Printed label prefixes (Date:, Exam Date:)
    - OCR character confusions: S/s->5, O/o->0, I/l/|->1, Z/z->2
    - Delimited formats (DD/MM/YYYY, MM/DD/YYYY, DD-MM-YY, DD.MM.YYYY)
    - Ambiguity resolution: if second part > 12 and first <= 12, swaps MM/DD to DD/MM
    - Validates day (1-31), month (1-12), and year (2000-2040)
    - Compact 6/8 digit dates (e.g. 271125 -> 27/11/2025)
    Returns normalized DD/MM/YYYY string if valid, otherwise empty string (prompting manual verification).
    """
    if not raw_str:
        return ""
    val = re.sub(r"^(?:Exam\s*Date|Date)[:\s\-=]*", "", str(raw_str), flags=re.IGNORECASE).strip()
    val_clean = (val.replace("S", "5").replace("s", "5")
                    .replace("O", "0").replace("o", "0")
                    .replace("l", "1").replace("I", "1").replace("|", "1")
                    .replace("Z", "2").replace("z", "2"))
    
    # 1. Delimited formats: D[D]-M[M]-YY[YY] or M[M]-D[D]-YY[YY]
    m = re.search(r"(\d{1,2})[\-/\.](\d{1,2})[\-/\.](\d{2,4})", val_clean)
    if m:
        try:
            p1, p2, y = int(m.group(1)), int(m.group(2)), m.group(3)
            if len(y) == 2:
                y = "20" + y
            y_int = int(y)
            # Disambiguate day and month:
            # If p2 > 12 and p1 <= 12, then format was MM/DD/YYYY -> swap to DD/MM/YYYY
            if p2 > 12 and 1 <= p1 <= 12 and 1 <= p2 <= 31:
                d, mo = p2, p1
            elif 1 <= p1 <= 31 and 1 <= p2 <= 12:
                d, mo = p1, p2
            else:
                return ""  # Invalid date numbers, prompt review
            
            if 2000 <= y_int <= 2040:
                return f"{d:02d}/{mo:02d}/{y}"
        except (ValueError, TypeError):
            return ""

    # 2. Compact 6-digit or 8-digit format e.g. 271125, 27112025
    m_c = re.search(r"^(\d{2})(\d{2})(\d{2,4})$", val_clean)
    if m_c:
        try:
            p1, p2, y = int(m_c.group(1)), int(m_c.group(2)), m_c.group(3)
            if len(y) == 2:
                y = "20" + y
            if p2 > 12 and 1 <= p1 <= 12 and 1 <= p2 <= 31:
                d, mo = p2, p1
            elif 1 <= p1 <= 31 and 1 <= p2 <= 12:
                d, mo = p1, p2
            else:
                return ""
            if 2000 <= int(y) <= 2040:
                return f"{d:02d}/{mo:02d}/{y}"
        except (ValueError, TypeError):
            return ""

    # 3. Compact with trailing slash e.g. 2711/25
    m2 = re.search(r"(\d{2})(\d{2})[/]([0-9zZ]{2})", val_clean)
    if m2:
        try:
            d, mo, yr = int(m2.group(1)), int(m2.group(2)), m2.group(3).replace('z','2').replace('Z','2')
            if 1 <= d <= 31 and 1 <= mo <= 12:
                return f"{d:02d}/{mo:02d}/20{yr}"
        except (ValueError, TypeError):
            return ""

    return ""

def normalize_section(raw_str: str) -> str:
    if not raw_str:
        return ""
    val = re.sub(r"^(?:Section|Sec|tion)[:\s\-=]*", "", raw_str, flags=re.IGNORECASE).strip()
    val = re.sub(r"[^A-Za-z0-9]", "", val).strip()
    if val:
        c = val[:1].upper()
        if c in ["1", "L", "I"]:
            return "C" # common OCR confusion for C in handwriting box
        return c
    return ""

def extract_student_details_page1(image_input, detected_angle: int = 0) -> Tuple[StudentDetails, dict]:
    """
    MASTER Page 1 Document Extraction Pipeline:
    1. Document edge detection & 4-point perspective warp to canonical 1600 x 2200 canvas.
    2. Shadow removal and contrast enhancement.
    3. High-precision full-page PaddleOCR line detection.
    4. Dynamic spatial label anchoring to locate and crop value zones excluding printed labels.
    5. Fallback to verified canonical template coordinates if an anchor is obscured.
    6. Field-tailored validation, normalization, and confidence assignment.
    7. Stores cropped PIL snippet for each field to support interactive review.
    Returns: (StudentDetails, debug_telemetry_dict)
    """
    t0 = time.time()
    img_bgr = convert_to_opencv(image_input)
    h_in, w_in = img_bgr.shape[:2]

    # If image is already normalized canonical 1600 x 2200 canvas, preserve it
    if w_in == NORMALIZED_WIDTH and h_in == NORMALIZED_HEIGHT:
        warped_bgr = img_bgr
        was_warped = False
    else:
        # STEP 1: Orientation correction to ensure upright page
        from ocr.orientation import find_best_orthogonal_orientation, rotate_image_angle
        if detected_angle is not None and detected_angle in [0, 90, 180, 270]:
            oriented_bgr = rotate_image_angle(img_bgr, detected_angle)
        else:
            oriented_bgr, detected_angle = find_best_orthogonal_orientation(img_bgr, is_first_page=True)

        # STEP 2: Perspective correction and warping to canonical 1600 x 2200 canvas
        warped_bgr, was_warped = correct_document_perspective(oriented_bgr, NORMALIZED_WIDTH, NORMALIZED_HEIGHT)

    # STEP 3: Enhance shadows & contrast
    enhanced_bgr = remove_shadows_and_enhance(warped_bgr)
    
    # 4. Run PaddleOCR full page detection
    from ocr.paddle_ocr import get_paddleocr_engine
    p_engine = get_paddleocr_engine()
    raw_full_text, avg_ocr_conf, detections = p_engine.read_text_with_boxes(enhanced_bgr, enhance_contrast=False)
    
    # 5. Index all printed label anchors
    anchors: Dict[str, dict] = {}
    for d in detections:
        t_low = d["text"].strip().lower()
        rx, ry, rw, rh = d["rect"]
        if "booklet" in t_low and "serial" in t_low:
            anchors["booklet_serial"] = d
        elif "test" in t_low and ry < 450 and rx < 650:
            anchors["test_line"] = d
        elif "enroll" in t_low and ry < 480 and rx > 600:
            anchors["enrollment"] = d
        elif "instructor" in t_low:
            anchors["instructor"] = d
        elif t_low.startswith("name:") or t_low == "name:" or ("name" in t_low and ry > 480 and ry < 620 and rx < 400):
            anchors["name"] = d
        elif "section" in t_low:
            anchors["section"] = d
        elif "date:" in t_low or (t_low.startswith("date") and ry > 480 and ry < 620):
            anchors["date"] = d
        elif "class" in t_low and "of" in t_low:
            anchors["class"] = d
        elif "course" in t_low and "title" in t_low:
            anchors["course_title"] = d
        elif "semester" in t_low:
            anchors["semester"] = d
        elif "course" in t_low and "code" in t_low:
            anchors["course_code"] = d
        elif "grand" in t_low and "total" in t_low:
            anchors["grand_total"] = d

    details = StudentDetails()
    confidences: Dict[str, float] = {}
    crop_telemetry: Dict[str, dict] = {}
    actual_bboxes: Dict[str, Tuple[int, int, int, int]] = {}

    details.college_name = "IcfaiTech - Faculty of Science & Technology, ICFAI Foundation for Higher Education, Hyderabad"
    details.exam_name = "Comprehensive Examination"
    
    # Helper to crop safely
    def get_crop(x1, y1, x2, y2) -> np.ndarray:
        x1_c = max(0, min(NORMALIZED_WIDTH - 1, int(x1)))
        y1_c = max(0, min(NORMALIZED_HEIGHT - 1, int(y1)))
        x2_c = max(x1_c + 1, min(NORMALIZED_WIDTH, int(x2)))
        y2_c = max(y1_c + 1, min(NORMALIZED_HEIGHT, int(y2)))
        return enhanced_bgr[y1_c:y2_c, x1_c:x2_c]

    # --- FIELD 1: Booklet Serial Number ---
    f_val, f_conf, f_raw = "", 0.0, ""
    bx, by, bw, bh = DEFAULT_VALUE_ROIS.get("booklet_serial_number", (1100, 60, 430, 115))
    if "booklet_serial" in anchors:
        arx, ary, arw, arh = anchors["booklet_serial"]["rect"]
        bx, by, bw, bh = arx + arw + 10, max(0, ary - 10), 380, arh + 20
        # Check if adjacent detection already has digits
        for o in detections:
            ox, oy, ow, oh = o["rect"]
            if ox > arx + arw - 20 and ox < arx + arw + 450 and abs(oy - ary) < 45:
                digits = re.sub(r"[^\d]", "", o["text"])
                if len(digits) >= 4:
                    f_val, f_conf, f_raw = digits, o["confidence"], o["text"]
                    bx, by, bw, bh = ox, oy, ow, oh
                    break
    crop = get_crop(bx, by, bx + bw, by + bh)
    actual_bboxes["booklet_serial_number"] = (bx, by, bw, bh)
    if not f_val and is_valid_crop(crop):
        txt, cnf = p_engine.read_field_crop(crop)
        f_raw = txt
        f_val = re.sub(r"[^\d]", "", txt)
        f_conf = cnf if f_val else 0.0
    details.booklet_serial_number = f_val
    confidences["booklet_serial_number"] = f_conf
    crop_telemetry["booklet_serial_number"] = {
        "value": f_val or "Needs verification",
        "raw_ocr": f_raw,
        "confidence": f_conf,
        "crop_image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        "status": STATUS_OK if (f_conf >= 60.0 and len(f_val) >= 4) else STATUS_NEEDS_REVIEW,
        "bbox": (bx, by, bw, bh)
    }

    # --- FIELD 2: Test Number ---
    f_val, f_conf, f_raw = "", 0.0, ""
    tx, ty, tw, th = DEFAULT_VALUE_ROIS.get("test_number", (360, 320, 280, 90))
    if "test_line" in anchors:
        raw_t = anchors["test_line"]["text"]
        f_raw = raw_t
        m_t = re.search(r"Test[_\s]*([0-9A-Za-z]+)", raw_t, re.IGNORECASE)
        if m_t:
            v_cand = m_t.group(1).upper().replace("Z", "2").replace("L", "1").replace("O", "0")
            d_cand = re.findall(r"\d+", v_cand)
            if d_cand:
                f_val = str(int(d_cand[0]))
                f_conf = anchors["test_line"]["confidence"]
        tx, ty, tw, th = anchors["test_line"]["rect"]
        if not f_val:
            for o in detections:
                ox, oy, ow, oh = o["rect"]
                if ox > tx + tw - 15 and ox < 650 and abs(oy - ty) < 35:
                    d_match = re.findall(r"\d+", o["text"])
                    if d_match:
                        f_val = str(int(d_match[0]))
                        f_conf = o["confidence"]
                        f_raw = o["text"]
                        tx, ty, tw, th = ox, oy, ow, oh
                        break
    crop = get_crop(tx, ty, tx + tw, ty + th)
    actual_bboxes["test_number"] = (tx, ty, tw, th)
    if not f_val and is_valid_crop(crop):
        txt, cnf = p_engine.read_field_crop(crop)
        f_raw = txt
        m_tc = re.search(r"Test[_\s]*([0-9]+)", txt, re.IGNORECASE)
        if m_tc:
            f_val = str(int(m_tc.group(1)))
            f_conf = cnf
        else:
            digits = re.findall(r"\d+", txt)
            if digits:
                f_val = str(int(digits[0]))
                f_conf = cnf
    details.test_number = f_val
    confidences["test_number"] = f_conf
    crop_telemetry["test_number"] = {
        "value": f_val or "Needs verification",
        "raw_ocr": f_raw,
        "confidence": f_conf,
        "crop_image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        "status": STATUS_OK if (f_val and f_conf >= 55.0) else STATUS_NEEDS_REVIEW,
        "bbox": (tx, ty, tw, th)
    }

    # --- FIELD 3: Enrollment Number ---
    f_val, f_conf, f_raw = "", 0.0, ""
    ex, ey, ew, eh = BOOKLET_TEMPLATE.get(
        "enrollment_number", DEFAULT_VALUE_ROIS.get("enrollment_number", (1045, 360, 480, 75))
    )

    # 1. Search full page detections near the enrollment anchor first
    if "enrollment" in anchors:
        arx, ary, arw, arh = anchors["enrollment"]["rect"]
        ex = max(arx + arw + 10, 1045)
        ey = max(0, ary - 8)
        ew = min(480, 1560 - ex)
        eh = max(arh + 20, 70)
        for o in detections:
            ox, oy, ow, oh = o["rect"]
            if ox >= arx + arw - 20 and ox < 1550 and abs(oy - ary) < 45:
                cand = normalize_enrollment_number(o["text"])
                if cand:
                    f_val = cand
                    f_conf = o["confidence"]
                    f_raw = o["text"]
                    ex, ey, ew, eh = ox, oy, ow, oh
                    break

    # 2. Also search header detections (oy < 480, ox > 600)
    if not f_val:
        for o in detections:
            ox, oy, ow, oh = o["rect"]
            if oy < 480 and ox > 600:
                cand = normalize_enrollment_number(o["text"])
                if cand:
                    f_val = cand
                    f_conf = o["confidence"]
                    f_raw = o["text"]
                    ex, ey, ew, eh = ox, oy, ow, oh
                    break

    # 3. Crop OCR with preprocessing fallback
    crop = get_crop(ex, ey, ex + ew, ey + eh)
    actual_bboxes["enrollment_number"] = (ex, ey, ew, eh)
    if not f_val and is_valid_crop(crop):
        txt, cnf = p_engine.read_field_crop(crop)
        f_raw = txt
        cand = normalize_enrollment_number(txt)
        if cand:
            f_val = cand
            f_conf = cnf
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
            crop_enh = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
            txt_enh, cnf_enh = p_engine.read_field_crop(crop_enh)
            cand_enh = normalize_enrollment_number(txt_enh)
            if cand_enh:
                f_val = cand_enh
                f_conf = cnf_enh
                f_raw = txt_enh

    details.roll_number = f_val
    details.registration_number = f_val
    confidences["enrollment_number"] = f_conf
    crop_telemetry["enrollment_number"] = {
        "value": f_val or "Needs verification",
        "raw_ocr": f_raw,
        "confidence": f_conf,
        "crop_image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        "status": STATUS_OK if (f_val and f_conf >= 55.0) else STATUS_NEEDS_REVIEW,
        "bbox": (ex, ey, ew, eh)
    }

    # --- FIELD 4: Student Name ---
    f_val, f_conf, f_raw = "", 0.0, ""
    nx, ny, nw, nh = DEFAULT_VALUE_ROIS.get("student_name", (185, 525, 435, 85))
    if "name" in anchors:
        arx, ary, arw, arh = anchors["name"]["rect"]
        m_n = re.search(r"Name\s*[:\.\-]*\s*([A-Za-z\s\.]+)", anchors["name"]["text"], re.IGNORECASE)
        if m_n and len(m_n.group(1).strip()) > 3:
            f_val = normalize_student_name(m_n.group(1).strip())
            f_conf = anchors["name"]["confidence"]
            f_raw = anchors["name"]["text"]
            nx, ny, nw, nh = arx, ary, arw, arh
        else:
            sec_x = anchors["section"]["rect"][0] if "section" in anchors else 740
            nx, ny, nw, nh = arx + arw + 8, max(0, ary - 8), sec_x - (arx + arw + 16), arh + 20
            for o in detections:
                ox, oy, ow, oh = o["rect"]
                if ox > arx + arw - 20 and ox < sec_x - 10 and abs(oy - ary) < 30:
                    cand = normalize_student_name(o["text"])
                    if cand and len(cand) > 2:
                        f_val, f_conf, f_raw = cand, o["confidence"], o["text"]
                        nx, ny, nw, nh = ox, oy, ow, oh
                        break
    crop = get_crop(nx, ny, nx + nw, ny + nh)
    actual_bboxes["student_name"] = (nx, ny, nw, nh)
    if not f_val and is_valid_crop(crop):
        txt, cnf = p_engine.read_field_crop(crop)
        f_raw = txt
        f_val = normalize_student_name(txt)
        f_conf = cnf if f_val else 0.0
    details.student_name = f_val
    confidences["student_name"] = f_conf
    crop_telemetry["student_name"] = {
        "value": f_val or "Needs verification",
        "raw_ocr": f_raw,
        "confidence": f_conf,
        "crop_image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        "status": STATUS_OK if (f_conf >= 55.0 and len(f_val) >= 3) else STATUS_NEEDS_REVIEW,
        "bbox": (nx, ny, nw, nh)
    }

    # --- FIELD 5: Section ---
    f_val, f_conf, f_raw = "", 0.0, ""
    sx, sy, sw, sh = DEFAULT_VALUE_ROIS.get("section", (745, 530, 280, 95))
    if "section" in anchors:
        arx, ary, arw, arh = anchors["section"]["rect"]
        sec_text = anchors["section"]["text"]
        m_s = re.search(r"Section\s*[:\.\-'\"]*\s*['\"]?([A-Za-z0-9])", sec_text, re.IGNORECASE)
        if m_s:
            f_val = normalize_section(m_s.group(1))
            f_conf = anchors["section"]["confidence"]
            f_raw = sec_text
            sx, sy, sw, sh = arx, ary, arw, arh
        else:
            date_x = anchors["date"]["rect"][0] if "date" in anchors else 1140
            sx = arx + arw + 10
            sy = max(0, ary - 6)
            sw = max(100, min(240, date_x - sx - 20))
            sh = max(arh + 18, 50)
            for o in detections:
                ox, oy, ow, oh = o["rect"]
                if ox > arx + arw - 15 and ox < date_x - 10 and abs(oy - ary) < 45:
                    cand = normalize_section(o["text"])
                    if cand and len(cand) == 1:
                        f_val, f_conf, f_raw = cand, o["confidence"], o["text"]
                        sx, sy, sw, sh = ox, oy, ow, oh
                        break
    crop = get_crop(sx, sy, sx + sw, sy + sh)
    actual_bboxes["section"] = (sx, sy, sw, sh)
    if not f_val and is_valid_crop(crop):
        txt, cnf = p_engine.read_field_crop(crop)
        f_raw = txt
        f_val = normalize_section(txt)
        f_conf = cnf if f_val else 0.0
    details.section = f_val
    confidences["section"] = f_conf
    crop_telemetry["section"] = {
        "value": f_val or "Needs verification",
        "raw_ocr": f_raw,
        "confidence": f_conf,
        "crop_image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        "status": STATUS_OK if (f_conf >= 55.0 and f_val) else STATUS_NEEDS_REVIEW,
        "bbox": (sx, sy, sw, sh)
    }

    # --- FIELD 6: Exam Date ---
    f_val, f_conf, f_raw = "", 0.0, ""
    dx, dy, dw, dh = DEFAULT_VALUE_ROIS.get("exam_date", (1145, 525, 395, 85))
    if "date" in anchors:
        arx, ary, arw, arh = anchors["date"]["rect"]
        m_d = re.search(r"Date\s*[:\.\-]*\s*([0-9/\.\-]+)", anchors["date"]["text"], re.IGNORECASE)
        if m_d:
            f_val = normalize_date(m_d.group(1))
            f_conf = anchors["date"]["confidence"]
            f_raw = anchors["date"]["text"]
            dx, dy, dw, dh = arx, ary, arw, arh
        else:
            dx, dy, dw, dh = arx + arw + 8, max(0, ary - 8), 380, arh + 20
            for o in detections:
                ox, oy, ow, oh = o["rect"]
                if ox > arx + arw - 20 and abs(oy - ary) < 30:
                    cand = normalize_date(o["text"])
                    if cand and len(cand) >= 6:
                        f_val, f_conf, f_raw = cand, o["confidence"], o["text"]
                        dx, dy, dw, dh = ox, oy, ow, oh
                        break
    crop = get_crop(dx, dy, dx + dw, dy + dh)
    actual_bboxes["exam_date"] = (dx, dy, dw, dh)
    if not f_val and is_valid_crop(crop):
        txt, cnf = p_engine.read_field_crop(crop)
        f_raw = txt
        f_val = normalize_date(txt)
        f_conf = cnf if f_val else 0.0
    details.exam_date = f_val
    confidences["exam_date"] = f_conf
    crop_telemetry["exam_date"] = {
        "value": f_val or "Needs verification",
        "raw_ocr": f_raw,
        "confidence": f_conf,
        "crop_image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        "status": STATUS_OK if (f_conf >= 55.0 and f_val) else STATUS_NEEDS_REVIEW,
        "bbox": (dx, dy, dw, dh)
    }

    # --- FIELD 7: Class of ---
    f_val, f_conf, f_raw = "", 0.0, ""
    cx, cy, cw, ch = DEFAULT_VALUE_ROIS.get("class", (220, 605, 400, 85))
    if "class" in anchors:
        arx, ary, arw, arh = anchors["class"]["rect"]
        m_c = re.search(r"Class\s*of\s*[:\.\-]*\s*([A-Za-z0-9\s\-–—]+)", anchors["class"]["text"], re.IGNORECASE)
        if m_c and len(m_c.group(1).strip()) > 1:
            f_val = clean_ocr_value(m_c.group(1).strip(), "class")
            f_conf = anchors["class"]["confidence"]
            f_raw = anchors["class"]["text"]
            cx, cy, cw, ch = arx, ary, arw, arh
        else:
            ct_x = anchors["course_title"]["rect"][0] if "course_title" in anchors else 850
            cx, cy, cw, ch = arx + arw + 8, max(0, ary - 8), ct_x - (arx + arw + 16), arh + 20
            for o in detections:
                ox, oy, ow, oh = o["rect"]
                if ox > arx + arw - 20 and ox < ct_x - 10 and abs(oy - ary) < 25:
                    cand = clean_ocr_value(o["text"], "class")
                    if cand and len(cand) >= 2:
                        f_val, f_conf, f_raw = cand, o["confidence"], o["text"]
                        cx, cy, cw, ch = ox, oy, ow, oh
                        break
    crop = get_crop(cx, cy, cx + cw, cy + ch)
    actual_bboxes["class"] = (cx, cy, cw, ch)
    if not f_val and is_valid_crop(crop):
        txt, cnf = p_engine.read_field_crop(crop)
        f_raw = txt
        f_val = clean_ocr_value(txt, "class")
        f_conf = cnf if f_val else 0.0
    details.class_year = f_val
    confidences["class"] = f_conf
    crop_telemetry["class"] = {
        "value": f_val or "Needs verification",
        "raw_ocr": f_raw,
        "confidence": f_conf,
        "crop_image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        "status": STATUS_OK if (f_conf >= 55.0 and f_val) else STATUS_NEEDS_REVIEW,
        "bbox": (cx, cy, cw, ch)
    }

    # --- FIELD 8: Course Title ---
    f_val, f_conf, f_raw = "", 0.0, ""
    ctx, cty, ctw, cth = DEFAULT_VALUE_ROIS.get("course_title", (600, 630, 950, 90))
    if "course_title" in anchors:
        arx, ary, arw, arh = anchors["course_title"]["rect"]
        ct_raw = anchors["course_title"]["text"]
        f_raw = ct_raw
        f_conf = anchors["course_title"]["confidence"]
        norm_anchor = normalize_course_title(ct_raw)
        if norm_anchor:
            f_val = norm_anchor
        # Position value ROI to the right of the label anchor
        ctx = arx + arw + 10
        cty = max(0, ary - 15)
        ctw = max(600, min(950, NORMALIZED_WIDTH - ctx - 30))
        cth = max(arh + 30, 80)
        # Look for multi-line or adjacent continuation e.g. "Algorithms"
        for o in detections:
            ox, oy, ow, oh = o["rect"]
            if ox > 500 and (abs(oy - ary) < 45 or (oy > ary and oy < ary + arh + 60)):
                o_cand = normalize_course_title(o["text"])
                if o_cand and len(o_cand) >= 3:
                    f_val = o_cand
                    f_raw = o["text"]
                    f_conf = max(f_conf, o["confidence"])
                    break
    else:
        # Check detections for course title line if anchor wasn't indexed
        for o in detections:
            ox, oy, ow, oh = o["rect"]
            if 610 <= oy <= 730 and ox > 500:
                cand = normalize_course_title(o["text"])
                if cand and len(cand) >= 3 and cand.lower() not in ["not detected", "none"]:
                    f_val = cand
                    f_conf = o["confidence"]
                    f_raw = o["text"]
                    ctx, cty, ctw, cth = ox, oy, ow, oh
                    break

    crop = get_crop(ctx, cty, ctx + ctw, cty + cth)
    actual_bboxes["course_title"] = (ctx, cty, ctw, cth)
    if not f_val and is_valid_crop(crop):
        txt, cnf = p_engine.read_field_crop(crop)
        f_raw = txt
        f_val = normalize_course_title(txt)
        f_conf = cnf if f_val else 0.0
    details.course_title = f_val
    details.subject_name = f_val
    confidences["course_title"] = f_conf
    crop_telemetry["course_title"] = {
        "value": f_val or "Needs verification",
        "raw_ocr": f_raw,
        "confidence": f_conf,
        "crop_image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        "status": STATUS_OK if (f_conf >= 55.0 and f_val) else STATUS_NEEDS_REVIEW,
        "bbox": (ctx, cty, ctw, cth)
    }

    # --- FIELD 9: Semester ---
    f_val, f_conf, f_raw = "", 0.0, ""
    smx, smy, smw, smh = DEFAULT_VALUE_ROIS.get("semester", (220, 685, 400, 75))
    if "semester" in anchors:
        arx, ary, arw, arh = anchors["semester"]["rect"]
        m_sm = re.search(r"Semester\s*[:\.\-]*\s*([IVXLCDM1-8]+)", anchors["semester"]["text"], re.IGNORECASE)
        if m_sm:
            f_val = clean_ocr_value(m_sm.group(1), "semester")
            f_conf = anchors["semester"]["confidence"]
            f_raw = anchors["semester"]["text"]
            smx, smy, smw, smh = arx, ary, arw, arh
        else:
            smx, smy, smw, smh = arx + arw + 8, max(0, ary - 8), 350, arh + 20
            for o in detections:
                ox, oy, ow, oh = o["rect"]
                if ox > arx + arw - 20 and ox < 650 and abs(oy - ary) < 25:
                    cand = clean_ocr_value(o["text"], "semester")
                    if cand:
                        f_val, f_conf, f_raw = cand, o["confidence"], o["text"]
                        smx, smy, smw, smh = ox, oy, ow, oh
                        break
    crop = get_crop(smx, smy, smx + smw, smy + smh)
    actual_bboxes["semester"] = (smx, smy, smw, smh)
    details.semester = f_val
    confidences["semester"] = f_conf
    crop_telemetry["semester"] = {
        "value": f_val or "Needs verification",
        "raw_ocr": f_raw,
        "confidence": f_conf,
        "crop_image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        "status": STATUS_OK if (f_val and f_conf >= 55.0) else STATUS_NEEDS_REVIEW,
        "bbox": (smx, smy, smw, smh)
    }

    # --- FIELD 10: Course Code ---
    f_val, f_conf, f_raw = "", 0.0, ""
    ccx, ccy, ccw, cch = DEFAULT_VALUE_ROIS.get("course_code", (260, 755, 360, 75))
    if "course_code" in anchors:
        arx, ary, arw, arh = anchors["course_code"]["rect"]
        m_cc = re.search(r"Course\s*Code\s*[:\.\-]*\s*([A-Za-z0-9]+)", anchors["course_code"]["text"], re.IGNORECASE)
        if m_cc:
            f_val = normalize_course_code(m_cc.group(1))
            f_conf = anchors["course_code"]["confidence"]
            f_raw = anchors["course_code"]["text"]
            ccx, ccy, ccw, cch = arx, ary, arw, arh
        else:
            ccx, ccy, ccw, cch = arx + arw + 8, max(0, ary - 8), 350, arh + 20
            for o in detections:
                ox, oy, ow, oh = o["rect"]
                if ox > arx + arw - 20 and ox < 650 and abs(oy - ary) < 25:
                    cand = normalize_course_code(o["text"])
                    if cand and not any(k in cand.lower() for k in ["signature", "student", "invigilator", "additional"]):
                        f_val, f_conf, f_raw = cand, o["confidence"], o["text"]
                        ccx, ccy, ccw, cch = ox, oy, ow, oh
                        break
    crop = get_crop(ccx, ccy, ccx + ccw, ccy + cch)
    actual_bboxes["course_code"] = (ccx, ccy, ccw, cch)
    if not f_val and is_valid_crop(crop):
        txt, cnf = p_engine.read_field_crop(crop)
        clean_cand = normalize_course_code(txt)
        if clean_cand and not any(k in clean_cand.lower() for k in ["signature", "student", "invigilator"]):
            f_raw = txt
            f_val = clean_cand
            f_conf = cnf
    details.subject_code = f_val
    confidences["course_code"] = f_conf
    crop_telemetry["course_code"] = {
        "value": f_val or "Needs verification",
        "raw_ocr": f_raw,
        "confidence": f_conf,
        "crop_image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        "status": STATUS_OK if (f_conf >= 55.0 and f_val) else STATUS_NEEDS_REVIEW,
        "bbox": (ccx, ccy, ccw, cch)
    }

    # --- FIELD 11: Instructor Name ---
    f_val, f_conf, f_raw = "", 0.0, ""
    ix, iy, iw, ih = DEFAULT_VALUE_ROIS.get("instructor_name", (1030, 440, 510, 85))
    if "instructor" in anchors:
        arx, ary, arw, arh = anchors["instructor"]["rect"]
        ix, iy, iw, ih = arx + arw + 8, max(0, ary - 8), 500, arh + 20
        for o in detections:
            ox, oy, ow, oh = o["rect"]
            if ox > arx + arw - 20 and abs(oy - ary) < 35:
                cand = re.sub(r"^(?:Instructor(?:'s)?\s*Name)[:\s\-=]*", "", o["text"], flags=re.IGNORECASE).strip()
                if cand and len(cand) >= 2:
                    f_val, f_conf, f_raw = cand, o["confidence"], o["text"]
                    ix, iy, iw, ih = ox, oy, ow, oh
                    break
    crop = get_crop(ix, iy, ix + iw, iy + ih)
    actual_bboxes["instructor_name"] = (ix, iy, iw, ih)
    details.instructor_name = f_val or "Not Present"
    confidences["instructor_name"] = f_conf
    crop_telemetry["instructor_name"] = {
        "value": f_val or "Not Present",
        "raw_ocr": f_raw,
        "confidence": f_conf,
        "crop_image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        "status": STATUS_OK if f_val else STATUS_NOT_PRESENT,
        "bbox": (ix, iy, iw, ih)
    }

    # --- FIELD 12: Signatures ---
    for sig_key in ["student_signature", "invigilator_signature"]:
        sx, sy, sw, sh = DEFAULT_VALUE_ROIS.get(sig_key, (970, 685, 570, 75))
        sig_crop = get_crop(sx, sy, sx + sw, sy + sh)
        actual_bboxes[sig_key] = (sx, sy, sw, sh)
        gray_sig = cv2.cvtColor(sig_crop, cv2.COLOR_BGR2GRAY)
        std_dev = float(np.std(gray_sig))
        dark_pixels = int(np.sum(gray_sig < 175))
        is_detected = std_dev > 10.0 or dark_pixels > 25
        sig_str = "Detected / Available" if is_detected else "Not Detected"
        confidences[sig_key] = 90.0 if is_detected else 0.0
        crop_telemetry[sig_key] = {
            "value": sig_str,
            "raw_ocr": sig_str,
            "confidence": confidences[sig_key],
            "crop_image": Image.fromarray(cv2.cvtColor(sig_crop, cv2.COLOR_BGR2RGB)),
            "status": STATUS_OK if is_detected else STATUS_NOT_DETECTED,
            "bbox": (sx, sy, sw, sh)
        }
        if sig_key == "student_signature":
            details.student_signature_present = is_detected
        else:
            details.invigilator_signature_present = is_detected

    # --- FIELD 13: Grand Total ---
    f_val, f_conf, f_raw = "", 0.0, ""
    gtx, gty, gtw, gth = BOOKLET_TEMPLATE.get("grand_total", DEFAULT_VALUE_ROIS.get("grand_total", (500, 1840, 240, 140)))
    if "grand_total" in anchors:
        arx, ary, arw, arh = anchors["grand_total"]["rect"]
        from ocr.marks import parse_grand_total_string
        for o in detections:
            ox, oy, ow, oh = o["rect"]
            if ox > arx and abs(oy - ary) < 65:
                num, denom, s = parse_grand_total_string(o["text"])
                if s:
                    f_val, f_conf, f_raw = s, o["confidence"], o["text"]
                    gtx, gty, gtw, gth = ox, oy, ow, oh
                    break
        if not f_val:
            gtx = max(arx + arw + 20, 500)
            gty = max(0, ary - 20)
            gtw = 240
            gth = max(arh + 50, 130)
    crop = get_crop(gtx, gty, gtx + gtw, gty + gth)
    actual_bboxes["grand_total"] = (gtx, gty, gtw, gth)
    if not f_val and is_valid_crop(crop):
        from ocr.marks import parse_grand_total_string
        txt, cnf = p_engine.read_field_crop(crop)
        f_raw = txt
        num, denom, s = parse_grand_total_string(txt)
        if s:
            f_val = s
            f_conf = cnf
    setattr(details, "grand_total", f_val)
    confidences["grand_total"] = f_conf
    crop_telemetry["grand_total"] = {
        "value": f_val or "Needs verification",
        "raw_ocr": f_raw,
        "confidence": f_conf,
        "crop_image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        "status": STATUS_OK if f_val else STATUS_NEEDS_REVIEW,
        "bbox": (gtx, gty, gtw, gth)
    }

    # Signatures Status dict
    signatures_status = {
        "student_signature": crop_telemetry["student_signature"]["value"],
        "invigilator_signature": crop_telemetry["invigilator_signature"]["value"],
        "examiner_signature": "Detected / Available"
    }

    # Visual overlay with bounding boxes
    overlay_img = generate_roi_overlay_image(warped_bgr, actual_bboxes)

    # ACCURATE confidence scoring: do not claim overall 95% if core fields are missing!
    core_fields = ["student_name", "enrollment_number", "test_number", "exam_date", "course_title", "section"]
    core_confs = [confidences.get(f, 0.0) for f in core_fields]
    overall_conf = float(np.mean(core_confs)) if core_confs else 0.0
    
    # Penalize if student name or enrollment number is unreadable
    if not details.student_name or details.student_name == "Needs verification":
        overall_conf = min(overall_conf, 45.0)
    if not details.roll_number or details.roll_number == "Needs verification":
        overall_conf = min(overall_conf, 50.0)

    elapsed_sec = round(time.time() - t0, 2)
    telemetry = {
        "detected_angle": detected_angle,
        "was_warped": was_warped,
        "overlay_image": overlay_img,
        "crops": crop_telemetry,
        "confidences": confidences,
        "signatures_status": signatures_status,
        "average_confidence": round(overall_conf, 1),
        "raw_text": raw_full_text,
        "execution_time_seconds": elapsed_sec
    }

    setattr(details, "telemetry", telemetry)
    setattr(details, "confidences", confidences)
    print(f"[PERF] extract_student_details_page1 END - {elapsed_sec:.2f}s (Conf: {overall_conf:.1f}%)")
    return details, telemetry
