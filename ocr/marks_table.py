"""
Marks Table OpenCV grid cell extraction and question-wise total calculation module.
Re-exports core functionality from ocr.marks to maintain a single source of truth.
"""

from ocr.marks import (
    check_tesseract,
    parse_mark_value,
    extract_red_ink_mask,
    is_cell_blank,
    ocr_cell_crop,
    extract_marks_table
)

__all__ = [
    "check_tesseract",
    "parse_mark_value",
    "extract_red_ink_mask",
    "is_cell_blank",
    "ocr_cell_crop",
    "extract_marks_table"
]
