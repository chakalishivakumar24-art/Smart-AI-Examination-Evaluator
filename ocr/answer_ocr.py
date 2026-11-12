import re
from typing import Dict, List, Tuple
from PIL import Image
from ocr.preprocessing import preprocess_image
from ocr.first_page_ocr import run_ocr, run_ocr_with_layout, run_handwriting_ocr_stack, run_ocr_engine_with_confidence

def run_page_ocr(pil_image: Image.Image) -> str:
    """Extracts text from an answer page image."""
    text, _ = run_ocr_engine_with_confidence(pil_image, max_width=1600)
    return text

def parse_questions_and_answers(full_text: str) -> Dict[str, str]:
    """
    Parses OCR text of answer pages and splits into a dictionary mapping question IDs (Q1, Q2...)
    to the student's answer text. Multi-page answer text is appended automatically.
    Supports formats: Q1, Q1., Q1:, Question 1, Ans 1, 1., 1), 1:.
    """
    answers = {}
    if not full_text:
        return answers

    # Pattern matching question headers like Q1, Q1., Q1:, Question 1, Ans 1, 1), 1., 1:
    pattern = r"(?:^|\n)\s*(?:Q(?:uestion|\.)?\s*(\d+)|Ans(?:wer)?\s*(\d+)|(\d+)[\.\:\)])\s*[:\-]?"
    matches = list(re.finditer(pattern, full_text, re.IGNORECASE))

    if not matches:
        # Fallback: if no question marker detected, treat text as Q1
        cleaned = full_text.strip()
        if cleaned:
            answers["Q1"] = cleaned
        else:
            answers["Q1"] = "[OCR could not confidently extract this answer. Please verify original sheet.]"
        return answers

    for i in range(len(matches)):
        m = matches[i]
        q_num = m.group(1) or m.group(2) or m.group(3)
        q_id = f"Q{q_num}"
        
        start_pos = m.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        
        answer_text = full_text[start_pos:end_pos].strip()
        if not answer_text or len(answer_text) < 2:
            answer_text = "[OCR could not confidently extract this answer. Please verify original sheet.]"

        if q_id in answers and answers[q_id] and answers[q_id] != "[OCR could not confidently extract this answer. Please verify original sheet.]":
            answers[q_id] += "\n" + answer_text
        else:
            answers[q_id] = answer_text

    return answers

def process_answer_page(image_input, engine_name: str = "paddleocr") -> Tuple[str, float]:
    """Preprocesses answer page image and extracts text using specified OCR Engine / YOLO pipeline."""
    if isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        pil_img = preprocess_image(image_input)

    from ocr.yolo_ocr import run_yolo_ocr_pipeline
    answer_text, confidence, _ = run_yolo_ocr_pipeline(pil_img, engine_name=engine_name)
    return answer_text, confidence

def process_answer_page_with_layout(image_input, engine_name: str = "paddleocr") -> Tuple[str, float, List[Tuple[int, int, int, int]]]:
    from ocr.yolo_ocr import run_yolo_ocr_pipeline
    answer_text, confidence, detections = run_yolo_ocr_pipeline(image_input, engine_name=engine_name)
    boxes = [(x, y, x + w, y + h) for x, y, w, h, _, _ in detections]
    return answer_text, confidence, boxes

import time

def extract_answers_from_pages(images: List[Image.Image], engine_name: str = "paddleocr") -> Tuple[Dict[str, str], str]:
    """
    Processes all answer page images (pages 2 to N), performs OCR using the selected engine,
    combines OCR text across pages, and returns question-segmented answers.
    Returns: (question_answers_dict, full_text_str)
    """
    t0 = time.time()
    print(f"[PERF] extract_answers_from_pages START (Engine: {engine_name})")
    combined_raw_text = []
    
    for idx, page_img in enumerate(images):
        page_text, confidence = process_answer_page(page_img, engine_name=engine_name)
        if page_text.strip():
            page_num_str = f"--- Page {idx + 2} ---\n" if len(images) > 1 else ""
            combined_raw_text.append(page_num_str + page_text)

    full_text_str = "\n\n".join(combined_raw_text)
    question_answers = parse_questions_and_answers(full_text_str)
    
    if not question_answers:
        question_answers = {"Q1": "[OCR could not confidently extract this answer. Please verify original sheet.]"}
        
    elapsed = time.time() - t0
    print(f"[PERF] extract_answers_from_pages END - {elapsed:.2f} sec")
    return question_answers, full_text_str


