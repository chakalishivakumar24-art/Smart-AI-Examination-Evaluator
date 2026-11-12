"""
Field validation & Signature ink presence detection module.
Enforces validation rules without inventing missing characters or silently replacing uncertain digits.
"""

import cv2
import re
import numpy as np
from typing import Tuple, Optional
from utils.constants import (
    STATUS_OK,
    STATUS_LOW_CONFIDENCE,
    STATUS_NEEDS_REVIEW,
    STATUS_NOT_DETECTED,
    STATUS_NOT_PRESENT
)

def detect_signature_presence(crop_bgr: np.ndarray) -> Tuple[bool, float, str]:
    """
    Analyzes ink stroke variance and pixel density to detect signature presence.
    Do NOT run OCR on signature ROIs.
    Returns (is_present, confidence, status_string).
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
        return False, 0.0, STATUS_NOT_DETECTED

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    std_dev = float(np.std(gray))
    dark_pixels = int(np.sum(gray < 175))

    is_present = std_dev > 11.0 or dark_pixels > 20
    status = STATUS_OK if is_present else STATUS_NOT_DETECTED
    conf = 90.0 if is_present else 0.0

    return is_present, conf, status

def validate_enrollment_number(raw_val: str) -> Tuple[bool, str, str]:
    """
    Validates enrollment number format:
    Preserves recognized letters and digits exactly.
    Returns: (is_valid, cleaned_val, message)
    """
    if not raw_val:
        return False, "", "Enrollment number is missing"
    val = re.sub(r"^(?:Enrollment|Registration|Roll|NO|NUM|N)[:\s\-=]*", "", raw_val, flags=re.IGNORECASE).strip()
    val = re.sub(r"[^\w]", "", val).upper()
    if re.match(r"^2[3-5][A-Z]{3,8}\d{4,8}$", val):
        return True, val, "Valid university enrollment format"
    return False, val, "Uncertain or non-standard format (Needs Verification)"

def validate_exam_date(raw_val: str) -> Tuple[bool, str, str]:
    """
    Validates complete examination date.
    Rejects incomplete single digits (like '3') or non-date strings.
    Returns: (is_valid, cleaned_date, message)
    """
    if not raw_val:
        return False, "", "Exam date is missing"
    val = re.sub(r"^(?:Date)[:\s\-=]*", "", raw_val, flags=re.IGNORECASE).strip()
    m = re.search(r"^(\d{1,2})[\-/\.](\d{1,2})[\-/\.](\d{2,4})$", val)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return True, val, "Valid exam date format"
    return False, val, "Incomplete or unreadable date (Needs Verification)"

def validate_course_code(raw_val: str) -> Tuple[bool, str, str]:
    """
    Validates course code format e.g. CS203, DS804, CS501.
    """
    if not raw_val:
        return False, "", "Course code is missing"
    val = re.sub(r"[^\w]", "", raw_val).upper()
    if re.match(r"^[A-Z]{2,4}\d{3,4}$", val):
        return True, val, "Valid course code format"
    return False, val, "Uncertain course code format (Needs Verification)"

def validate_semester(raw_val: str) -> Tuple[bool, str, str]:
    """
    Validates semester (Roman numerals I to VIII or numbers 1 to 8).
    """
    if not raw_val:
        return False, "", "Semester is missing"
    val = raw_val.strip().upper()
    if val in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "1", "2", "3", "4", "5", "6", "7", "8"]:
        return True, val, "Valid semester"
    return False, val, "Uncertain semester value (Needs Verification)"

def validate_grand_total(raw_val: str, calculated_sum: Optional[float] = None) -> Tuple[bool, str, str]:
    """
    Validates grand total. Rejects labels e.g. 'Examiner Signature:'.
    Optionally compares against calculated sum of marks when available.
    """
    if not raw_val:
        return False, "", "Grand total is missing"
    val = raw_val.strip()
    if any(k in val.lower() for k in ["examiner", "signature", "remarks", "pto"]):
        return False, "", "Confused with Examiner Signature label (Needs Verification)"

    m_den = re.search(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$", val)
    if m_den:
        obtained = float(m_den.group(1))
        denom = float(m_den.group(2))
        if calculated_sum is not None and abs(obtained - calculated_sum) > 0.1:
            return False, val, f"Written total ({obtained:g}) differs from question marks sum ({calculated_sum:g})"
        return True, val, "Valid total with maximum marks"

    try:
        f_val = float(val)
        if calculated_sum is not None and abs(f_val - calculated_sum) > 0.1:
            return False, val, f"Written total ({f_val:g}) differs from question marks sum ({calculated_sum:g})"
        return True, val, "Valid numerical total"
    except ValueError:
        return False, val, "Non-numeric total value (Needs Verification)"

def determine_field_extraction_status(field_name: str, value: str, confidence: float) -> str:
    """
    Determines extraction status code for a field crop.
    """
    val = str(value).strip() if value else ""
    if confidence >= 55.0 and val and val.lower() != "needs verification":
        return STATUS_OK
    elif val and val.lower() != "needs verification":
        return STATUS_LOW_CONFIDENCE
    else:
        if field_name in ["department", "examiner_remarks", "recheck_request"]:
            return STATUS_NOT_PRESENT
        return STATUS_NEEDS_REVIEW
