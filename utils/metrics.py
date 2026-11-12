"""
Metrics calculation module for cover-page metadata extraction.
Computes dynamic statistics for all 11 fields:
- Verified Correct
- Incorrect
- Needs Verification

Formulas:
  Accuracy (%) = (Number of Verified Correct Fields / 11) * 100
  Error (%) = (Number of Confirmed Incorrect Fields / 11) * 100
  Pending Verification (%) = (Number of Fields Needing Verification / 11) * 100
"""

import re
from typing import List, Dict, Any, Optional

COVER_FIELDS = [
    {"id": "student_name", "label": "Student Name", "placeholder": "Enter student name manually"},
    {"id": "enrollment_no", "label": "Enrollment No", "placeholder": "e.g. 23STUCHH010066"},
    {"id": "test_no", "label": "Test Number", "placeholder": "e.g. Test 1"},
    {"id": "section", "label": "Section", "placeholder": "e.g. C"},
    {"id": "date", "label": "Exam Date", "placeholder": "e.g. 27/11/2025"},
    {"id": "class", "label": "Class / Batch", "placeholder": "e.g. 2023 - 27"},
    {"id": "course_title", "label": "Course Title", "placeholder": "e.g. Machine Learning"},
    {"id": "course_code", "label": "Course Code", "placeholder": "e.g. DS804"},
    {"id": "semester", "label": "Semester", "placeholder": "e.g. V"},
    {"id": "instructor_name", "label": "Instructor Name", "placeholder": "e.g. Dr. A. Sharma"},
    {"id": "grand_total", "label": "Grand Total", "placeholder": "e.g. 5.5 or 18/30"}
]


def get_student_field_value(student_info, field_id: str) -> str:
    """Extract a cleaned string value for a given field ID from a StudentDetails model object."""
    if not student_info:
        return ""
    if field_id == "enrollment_no":
        v = getattr(student_info, "roll_number", None) or getattr(student_info, "registration_number", None)
    elif field_id == "course_title":
        v = getattr(student_info, "course_title", None) or getattr(student_info, "subject_name", None)
    elif field_id == "grand_total":
        v = getattr(student_info, "grand_total", None)
    elif field_id == "instructor_name":
        v = getattr(student_info, "instructor_name", None)
    elif field_id == "test_no":
        v = getattr(student_info, "test_number", None)
    elif field_id == "date":
        v = getattr(student_info, "exam_date", None)
    elif field_id == "class":
        v = getattr(student_info, "class_year", None)
    elif field_id == "course_code":
        v = getattr(student_info, "subject_code", None)
    elif field_id == "semester":
        v = getattr(student_info, "semester", None)
    elif field_id == "section":
        v = getattr(student_info, "section", None)
    elif field_id == "student_name":
        v = getattr(student_info, "student_name", None)
    else:
        v = getattr(student_info, field_id, None)

    if v and str(v).strip() and str(v).strip().lower() not in ["not detected", "not_detected", "none", "needs review"]:
        return str(v).strip()
    return ""


def update_student_details_field(student_info, field_id: str, new_val: str):
    """Updates the corresponding field in the StudentDetails model object."""
    if not student_info:
        return
    clean_val = str(new_val).strip() if new_val is not None else ""
    if field_id == "student_name":
        student_info.student_name = clean_val
    elif field_id == "enrollment_no":
        student_info.roll_number = clean_val
        student_info.registration_number = clean_val
    elif field_id == "test_no":
        student_info.test_number = clean_val
    elif field_id == "section":
        student_info.section = clean_val
    elif field_id == "date":
        student_info.exam_date = clean_val
    elif field_id == "class":
        student_info.class_year = clean_val
    elif field_id == "course_title":
        student_info.course_title = clean_val
        student_info.subject_name = clean_val
    elif field_id == "course_code":
        student_info.subject_code = clean_val
    elif field_id == "semester":
        student_info.semester = clean_val
    elif field_id == "instructor_name":
        setattr(student_info, "instructor_name", clean_val)
    elif field_id == "grand_total":
        setattr(student_info, "grand_total", clean_val)
    else:
        setattr(student_info, field_id, clean_val)


def _clean_date_str(val: str) -> str:
    """Helper for date normalization and comparison across delimiter formats."""
    if not val:
        return ""
    v = re.sub(r"^(?:Exam\s*Date|Date)[:\s\-=]*", "", str(val), flags=re.IGNORECASE).strip()
    v = (v.replace("S", "5").replace("s", "5")
           .replace("O", "0").replace("o", "0")
           .replace("l", "1").replace("I", "1").replace("|", "1")
           .replace("Z", "2").replace("z", "2"))
    m = re.search(r"(\d{1,2})[\-/\.](\d{1,2})[\-/\.](\d{2,4})", v)
    if m:
        p1, p2, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if len(y) == 2:
            y = "20" + y
        if p2 > 12 and 1 <= p1 <= 12 and 1 <= p2 <= 31:
            d, mo = p2, p1
        elif 1 <= p1 <= 31 and 1 <= p2 <= 12:
            d, mo = p1, p2
        else:
            return re.sub(r"[\.\-/\s]+", "/", v)
        return f"{d:02d}/{mo:02d}/{y}"
    return re.sub(r"[\.\-/\s]+", "/", v)


def compare_field_values(raw_extracted: str, ground_truth: str, field_id: str) -> bool:
    """
    Compare raw OCR extraction with verified ground truth value.
    Handles field-specific normalizations:
    - Enrollment Number: case and non-alphanumeric insensitive
    - Test Number: numeric digit extraction comparison (e.g. '1' vs 'Test 1')
    - Class / Batch: hyphen spacing insensitive (e.g. '2023 - 27' vs '2023-27')
    - Grand Total: direct match or numerator/float comparison (e.g. '18/30' vs 18.0)
    - Date: normalized slashes and hyphens with DD/MM/YYYY disambiguation
    """
    clean_gt = (ground_truth or "").strip()
    clean_ext = (raw_extracted or "").strip()
    if not clean_gt:
        return False
    if not clean_ext or clean_ext.lower() in ["not detected", "not_detected", "none", "needs review", "not present"]:
        return False

    if field_id == "enrollment_no":
        return re.sub(r"[^A-Za-z0-9]", "", clean_ext).upper() == re.sub(r"[^A-Za-z0-9]", "", clean_gt).upper()
    elif field_id == "test_no":
        d_ext = re.findall(r"\d+", clean_ext)
        d_gt = re.findall(r"\d+", clean_gt)
        if d_ext and d_gt:
            return int(d_ext[0]) == int(d_gt[0])
        return re.sub(r"\s+", "", clean_ext).lower() == re.sub(r"\s+", "", clean_gt).lower()
    elif field_id == "class":
        c_ext = re.sub(r"\s*-\s*", "-", clean_ext).lower()
        c_gt = re.sub(r"\s*-\s*", "-", clean_gt).lower()
        return c_ext == c_gt
    elif field_id == "grand_total":
        if re.sub(r"\s+", "", clean_ext).lower() == re.sub(r"\s+", "", clean_gt).lower():
            return True
        try:
            v_ext = float(clean_ext.split("/")[0].strip()) if "/" in clean_ext else float(clean_ext)
            v_gt = float(clean_gt.split("/")[0].strip()) if "/" in clean_gt else float(clean_gt)
            return abs(v_ext - v_gt) < 1e-3
        except Exception:
            return False
    elif field_id == "date":
        return _clean_date_str(clean_ext) == _clean_date_str(clean_gt)
    else:
        return re.sub(r"\s+", " ", clean_ext).strip().lower() == re.sub(r"\s+", " ", clean_gt).strip().lower()


def calculate_cover_field_metrics(
    cover_fields: List[Dict[str, Any]],
    v_map: Dict[str, Dict[str, Any]],
    initial_extracted_fields: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate dynamic statistics for all cover page fields.
    Each field has one of three mutually exclusive statuses:
      - 'Verified Correct'
      - 'Incorrect'
      - 'Needs Verification'

    Formulas:
      Accuracy (%) = (Number of Verified Correct Fields / 11) * 100
      Error (%) = (Number of Confirmed Incorrect Fields / 11) * 100
      Pending Verification (%) = (Number of Fields Needing Verification / 11) * 100
      Verification Completion Rate (%) = (Reviewed Fields / 11) * 100

    Guarantees:
      - Verified Correct + Incorrect + Needs Verification == total_fields (11)
      - Missing or undetected values are never counted as verified correct.
      - Zero fields handled safely without division-by-zero errors.
    """
    total_fields = len(cover_fields)
    verified_correct_count = 0
    incorrect_count = 0
    needs_verification_count = 0
    field_status_map = {}

    for f in cover_fields:
        fid = f["id"]
        field_state = v_map.get(fid, {}) if v_map else {}
        explicit_status = field_state.get("status")
        is_ver = bool(field_state.get("verified", False))
        is_explicit_incorrect = bool(field_state.get("is_explicit_incorrect", False))
        gt_val = field_state.get("ground_truth", "")
        ocr_val = initial_extracted_fields.get(fid, "") if initial_extracted_fields else ""

        is_ocr_missing = (not ocr_val) or str(ocr_val).strip().lower() in [
            "not detected", "not_detected", "none", "needs review", "not present", "not_present", ""
        ]

        if explicit_status in ["Verified", "Verified Correct"]:
            status = "Verified Correct"
            verified_correct_count += 1
        elif explicit_status == "Incorrect" or is_explicit_incorrect:
            status = "Incorrect"
            incorrect_count += 1
        elif explicit_status in ["Needs Verification", "Needs Manual Verification", "Pending"]:
            status = "Needs Verification"
            needs_verification_count += 1
        elif not is_ver:
            status = "Needs Verification"
            needs_verification_count += 1
        elif (not is_ocr_missing) and compare_field_values(ocr_val, gt_val, fid):
            status = "Verified Correct"
            verified_correct_count += 1
        else:
            status = "Incorrect"
            incorrect_count += 1

        field_status_map[fid] = status

    total_detected_count = sum(
        1 for f in cover_fields 
        if initial_extracted_fields.get(f["id"]) and str(initial_extracted_fields.get(f["id"])).strip().lower() not in [
            "not detected", "not_detected", "none", "needs review", "not present", "not_present", "unreadable", "—", "-", "n/a", "na", ""
        ]
    )

    if total_fields == 0:
        accuracy_pct = 0.0
        error_pct = 0.0
        pending_pct = 0.0
        completion_rate_pct = 0.0
        reviewed_accuracy_pct = 0.0
        accuracy_str = "0.00%"
        error_str = "0.00%"
        pending_str = "0.00%"
        completion_rate_str = "0.00%"
        reviewed_accuracy_str = "N/A"
    else:
        accuracy_pct = round((verified_correct_count / total_fields) * 100.0, 2)
        error_pct = round((incorrect_count / total_fields) * 100.0, 2)
        pending_pct = round((needs_verification_count / total_fields) * 100.0, 2)
        reviewed_count = verified_correct_count + incorrect_count
        completion_rate_pct = round((reviewed_count / total_fields) * 100.0, 2)
        reviewed_accuracy_pct = round((verified_correct_count / reviewed_count) * 100.0, 2) if reviewed_count > 0 else 0.0

        accuracy_str = f"{accuracy_pct:.2f}%"
        error_str = f"{error_pct:.2f}%"
        pending_str = f"{pending_pct:.2f}%"
        completion_rate_str = f"{completion_rate_pct:.2f}%"
        reviewed_accuracy_str = f"{reviewed_accuracy_pct:.2f}%" if reviewed_count > 0 else "N/A"

    reviewed_count = verified_correct_count + incorrect_count

    return {
        "total_fields": total_fields,
        "total_detected_count": total_detected_count,
        "auto_verified_count": verified_correct_count,
        "verified_correct_count": verified_correct_count,
        "incorrect_count": incorrect_count,
        "needs_verification_count": needs_verification_count,
        "reviewed_count": reviewed_count,
        "completion_rate_pct": completion_rate_pct,
        "completion_rate_str": completion_rate_str,
        "reviewed_accuracy_pct": reviewed_accuracy_pct,
        "reviewed_accuracy_str": reviewed_accuracy_str,
        "field_status_map": field_status_map,
        "accuracy_pct": accuracy_pct,
        "error_pct": error_pct,
        "pending_pct": pending_pct,
        "accuracy_str": accuracy_str,
        "error_str": error_str,
        "pending_str": pending_str,
        "accuracy_sub": f"{verified_correct_count} of {total_fields} fields verified correct",
        "error_sub": f"{incorrect_count} of {total_fields} fields confirmed incorrect",
        "pending_sub": f"{needs_verification_count} of {total_fields} fields need verification",
        "completion_sub": f"{reviewed_count} of {total_fields} fields reviewed ({completion_rate_str})",
    }
