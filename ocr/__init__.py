# OCR Package
from ocr.orientation import (
    find_best_orthogonal_orientation,
    detect_and_rotate_image,
    detect_orientation_angle,
    rotate_image_angle,
    transform_bbox,
    inverse_transform_bbox
)
from ocr.preprocessing import preprocess_image, preprocess_image_for_ocr, convert_to_opencv, detect_form_field_regions
from ocr.template import generate_roi_overlay_image, normalize_to_canvas
from ocr.handwriting import extract_field_crop_multipass
from ocr.validation import detect_signature_presence, determine_field_extraction_status
from ocr.marks import extract_marks_table, ocr_cell_crop, is_cell_blank, parse_mark_value
from ocr.student_details import extract_student_details_page1
from ocr.pipeline import process_first_page, process_uploaded_booklet
from ocr.answer_ocr import (
    process_answer_page,
    process_answer_page_with_layout,
    extract_answers_from_pages,
    parse_questions_and_answers
)
from ocr.yolo_ocr import run_yolo_ocr_pipeline, draw_yolo_detections, get_yolo_detector, get_easyocr_engine
from ocr.paddle_ocr import PaddleOCREngine, get_paddleocr_engine, draw_paddle_detections

__all__ = [
    "find_best_orthogonal_orientation",
    "detect_and_rotate_image",
    "preprocess_image",
    "preprocess_image_for_ocr",
    "convert_to_opencv",
    "detect_form_field_regions",
    "generate_roi_overlay_image",
    "normalize_to_canvas",
    "extract_field_crop_multipass",
    "detect_signature_presence",
    "determine_field_extraction_status",
    "extract_marks_table",
    "ocr_cell_crop",
    "is_cell_blank",
    "parse_mark_value",
    "extract_student_details_page1",
    "process_first_page",
    "process_uploaded_booklet",
    "process_answer_page",
    "process_answer_page_with_layout",
    "extract_answers_from_pages",
    "parse_questions_and_answers",
    "run_yolo_ocr_pipeline",
    "draw_yolo_detections",
    "get_yolo_detector",
    "get_easyocr_engine",
    "PaddleOCREngine",
    "get_paddleocr_engine"
]

