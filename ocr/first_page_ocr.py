import re
import os
import time
import cv2
from typing import Dict, Tuple, List
import numpy as np
from PIL import Image
from models.database_models import StudentDetails
from utils.text_parser import clean_ocr_value
from ocr.preprocessing import preprocess_image, preprocess_image_for_ocr, convert_to_opencv, detect_form_field_regions, find_best_orthogonal_orientation

_TESSERACT_AVAILABLE = True
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
            break
except ImportError:
    _TESSERACT_AVAILABLE = False

def check_tesseract_availability():
    if not _TESSERACT_AVAILABLE:
        raise RuntimeError("Tesseract OCR is not installed. Please install Tesseract OCR and restart the application.")

def resize_for_ocr(image: Image.Image, max_width: int = 1600) -> Image.Image:
    """Resizes a PIL Image to a maximum width for fast OCR while maintaining aspect ratio."""
    if not isinstance(image, Image.Image):
        return image
    w, h = image.size
    if w <= max_width:
        return image
    ratio = max_width / float(w)
    new_size = (max_width, int(h * ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)

def run_ocr_engine_with_confidence(pil_image: Image.Image, max_width: int = 1600) -> Tuple[str, float]:
    """
    Executes Tesseract OCR on a resized PIL Image.
    Returns (text, confidence) tuple of 2 values.
    """
    check_tesseract_availability()
    ocr_pil = resize_for_ocr(pil_image, max_width=max_width)
    
    try:
        data = pytesseract.image_to_data(ocr_pil, output_type=pytesseract.Output.DICT)
        text_lines = [str(t).strip() for t in data.get('text', []) if str(t).strip() != '']
        confs = []
        for c in data.get('conf', []):
            try:
                c_val = float(c)
                if c_val > 0:
                    confs.append(c_val)
            except (ValueError, TypeError):
                pass
        full_text = "\n".join(text_lines)
        avg_conf = float(np.mean(confs)) if confs else (85.0 if full_text.strip() else 0.0)
        return full_text, round(avg_conf, 1)
    except Exception as e:
        print(f"[Tesseract Error] {e}")
        return "", 0.0

def run_ocr(image_input) -> Tuple[str, float]:
    if isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        pil_img = preprocess_image(image_input, auto_rotate=False)
    return run_ocr_engine_with_confidence(pil_img)

def run_handwriting_ocr_stack(pil_image: Image.Image) -> str:
    text, _ = run_ocr_engine_with_confidence(pil_image)
    return text


def run_ocr_with_layout(image_input) -> Tuple[str, float, List[Tuple[int, int, int, int]]]:
    img_bgr = convert_to_opencv(image_input)
    boxes = detect_form_field_regions(img_bgr)
    text, confidence = run_ocr(image_input)
    return text, confidence, boxes

def extract_field_by_spatial_label(
    small_bgr: np.ndarray,
    data_df,
    label_keywords: List[str],
    field_name: str,
    psm_config: str = "--psm 7"
) -> Tuple[str, float]:
    """
    Locates printed field label coordinates in image_to_data DataFrame,
    crops the value area (to the right or immediately below), preprocesses crop,
    runs field-tailored Tesseract OCR, and applies confidence thresholding (> 45%).
    """
    check_tesseract_availability()
    if data_df is None or data_df.empty:
        return "", 0.0

    img_h, img_w = small_bgr.shape[:2]
    matched_box = None

    for _, row in data_df.iterrows():
        txt = str(row['text']).strip().lower()
        for kw in label_keywords:
            if kw.lower() in txt:
                matched_box = (int(row['left']), int(row['top']), int(row['width']), int(row['height']))
                break
        if matched_box:
            break

    if not matched_box:
        return "", 0.0

    lx, ly, lw, lh = matched_box

    # Candidate 1: Right region of label
    rx = lx + lw
    ry = max(0, ly - 8)
    rw = int(img_w * 0.40)
    rh = lh + 30
    
    crop = small_bgr[ry:min(img_h, ry+rh), rx:min(img_w, rx+rw)]
    
    # Fallback to Below region if Right region is tiny
    if crop.size == 0 or crop.shape[1] < 20:
        bx = max(0, lx - 10)
        by = ly + lh
        bw = max(lw * 2, int(img_w * 0.45))
        bh = int(lh * 2.5)
        crop = small_bgr[by:min(img_h, by+bh), bx:min(img_w, bx+bw)]

    if crop.size == 0:
        return "", 0.0

    # Preprocess Crop: Grayscale + CLAHE + Adaptive Threshold
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    crop_pil = Image.fromarray(enhanced)

    try:
        c_dict = pytesseract.image_to_data(crop_pil, config=psm_config, output_type=pytesseract.Output.DICT)
        raw_words = []
        valid_confs = []
        for t, c in zip(c_dict.get('text', []), c_dict.get('conf', [])):
            st_t = str(t).strip()
            if st_t:
                raw_words.append(st_t)
                try:
                    c_val = float(c)
                    if c_val > 0:
                        valid_confs.append(c_val)
                except (ValueError, TypeError):
                    pass

        if not raw_words:
            return "", 0.0

        raw_txt = " ".join(raw_words).strip()
        avg_conf = float(np.mean(valid_confs)) if valid_confs else 0.0

        cleaned_txt = clean_ocr_value(raw_txt, field_name)
        
        # Strict Confidence Thresholding: Reject low confidence or garbage single characters
        if avg_conf < 45.0 or len(cleaned_txt) < 2 and not cleaned_txt.isdigit():
            return "", 0.0

        return cleaned_txt, round(avg_conf, 1)
    except Exception as e:
        print(f"[Field Crop OCR Error] {field_name}: {e}")
        return "", 0.0

from ocr.student_details import extract_student_details_page1

def process_first_page(image_input, detected_angle: int = 0) -> Tuple[StudentDetails, str, float]:
    """
    Standardized Page 1 Processing API Function:
    Delegates to dedicated template ROI OCR module ocr.student_details.extract_student_details_page1.
    Returns: (student_details, raw_text, confidence)
    """
    t0 = time.time()
    print("[PERF] process_first_page START")
    details, telemetry = extract_student_details_page1(image_input, detected_angle=detected_angle)
    raw_text = telemetry.get("raw_text", "")
    overall_conf = telemetry.get("average_confidence", 85.0)
    
    # Attach telemetry dict as attribute for UI debug rendering
    setattr(details, "telemetry", telemetry)
    elapsed = time.time() - t0
    print(f"[PERF] process_first_page END - {elapsed:.2f} sec")
    return details, raw_text, overall_conf



def extract_field_regex_fallback(raw_text: str, field_name: str) -> Tuple[str, float]:
    """Fallback regex extractor for Page 1 text."""
    patterns = {
        "student_name": [r"(?:NAME|STUDENT)\s*[:\-=]*\s*([^\n\r]+)"],
        "roll_number": [r"(?:ROLL|HT|TICKET)\s*(?:NO|NUMBER)?\s*[:\-=]*\s*([A-Za-z0-9\-]+)"],
        "registration_number": [r"(?:REGISTRATION|REG)\s*(?:NO|NUMBER)?\s*[:\-=]*\s*([A-Za-z0-9\-]+)"],
        "subject_name": [r"(?:COURSE\s*TITLE|SUBJECT|PAPER)\s*[:\-=]*\s*([^\n\r]+)"],
        "subject_code": [r"(?:CODE)\s*[:\-=]*\s*([A-Za-z0-9\-]+)", r"\b([A-Z]{2,4}\s*\d{3,4})\b"],
        "exam_name": [r"(?:EXAM|EXAMINATION|TEST)\s*[:\-=]*\s*([^\n\r]+)"],
        "exam_date": [r"(?:DATE)\s*[:\-=]*\s*(\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4})"],
        "college_name": [r"(?:FACULTY|COLLEGE|UNIVERSITY|INSTITUTION)\s*[:\-=]*\s*([^\n\r]+)"],
        "semester": [r"(?:SEMESTER|SEM)\s*[:\-=]*\s*([IVXLCDM\d]+)"],
        "branch": [r"(?:DEPARTMENT|BRANCH|SECTION|CLASS)\s*[:\-=]*\s*([^\n\r]+)"]
    }
    pats = patterns.get(field_name, [])
    for p in pats:
        m = re.search(p, raw_text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            cleaned = clean_ocr_value(val, field_name)
            if cleaned and len(cleaned) >= 2:
                return cleaned, 85.0
    return "", 0.0

def extract_student_details(text: str, boxes=None) -> StudentDetails:
    student_details, _ = extract_student_details_from_text(text)
    return student_details

def extract_student_details_from_text(raw_text: str) -> Tuple[StudentDetails, Dict[str, float]]:
    details = StudentDetails()
    confidences = {}
    for f in ["student_name", "roll_number", "registration_number", "subject_name", "subject_code", "exam_name", "exam_date", "college_name", "semester", "branch"]:
        val, conf = extract_field_regex_fallback(raw_text, f)
        setattr(details, f, val)
        confidences[f] = conf
    return details, confidences

def extract_first_page_metadata(image_input) -> Tuple[StudentDetails, Dict[str, float], str, int]:
    student_details, combined_raw_text, _ = process_first_page(image_input)
    confidences = getattr(student_details, "confidences", {})
    best_angle = getattr(student_details, "detected_angle", 0)
    return student_details, confidences, combined_raw_text, best_angle
