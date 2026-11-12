"""
ICFAI Answer Booklet 7-Step Evaluation Pipeline Coordinator Module.
"""

import time
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Optional
from models.database_models import StudentDetails
from utils.pdf_processor import read_scanned_booklet
from ocr.preprocessing import find_best_orthogonal_orientation, convert_to_opencv
from ocr.student_details import extract_student_details_page1
from ocr.marks import extract_marks_table
from ocr.answer_ocr import extract_answers_from_pages
from evaluation.evaluator import evaluate_student_booklet
from evaluation.grading import determine_grade_and_status
from utils.sample_generator import seed_sample_model_answers

def process_first_page(image_input, detected_angle: int = 0) -> Tuple[StudentDetails, str, float]:
    """
    Page 1 Processing Function:
    Delegates to dedicated template ROI OCR module ocr.student_details.extract_student_details_page1.
    Returns: (student_details, raw_text, confidence)
    """
    t0 = time.time()
    print("[PERF] process_first_page START")
    details, telemetry = extract_student_details_page1(image_input, detected_angle=detected_angle)
    raw_text = telemetry.get("raw_text", "")
    overall_conf = telemetry.get("average_confidence", 85.0)
    
    setattr(details, "telemetry", telemetry)
    elapsed = time.time() - t0
    print(f"[PERF] process_first_page END - {elapsed:.2f} sec")
    return details, raw_text, overall_conf

def process_uploaded_booklet(
    uploaded_file,
    db,
    progress_box=None,
    progress_bar=None,
    manual_rotation_angle: Optional[int] = None
) -> dict:
    """
    Complete 7-Step Automated Booklet Evaluation Pipeline:
    1. Read PDF/Image file bytes
    2. Page Orientation & Normalization (1600 x 2200 canvas)
    3. Student & Exam Details OCR (ICFAI Master Booklet Fixed Template)
    4. Marks Table Grid Extraction & Question-Wise Marks Calculation
    5. Answer Extraction
    6. Evaluation & Grade Assignment
    7. Database Persistence & Result Structuring
    """
    pipeline_t0 = time.time()

    # STEP 1: READ BOOKLET
    t1_start = time.time()
    if progress_box and progress_bar:
        progress_box.info("📖 Step 1/7: Reading and validating uploaded booklet...")
        progress_bar.progress(14)
    images = read_scanned_booklet(uploaded_file)
    t1_elapsed = time.time() - t1_start

    # STEP 2: ORIENTATION & NORMALIZATION
    t2_start = time.time()
    if progress_box and progress_bar:
        progress_box.info("🔄 Step 2/7: Detecting page orientation & normalizing resolution...")
        progress_bar.progress(28)

    oriented_images = []
    best_angles = []
    for idx, img in enumerate(images):
        img_bgr = convert_to_opencv(img)
        if manual_rotation_angle is not None and manual_rotation_angle in [0, 90, 180, 270]:
            from ocr.orientation import rotate_image_angle
            best_bgr = rotate_image_angle(img_bgr, manual_rotation_angle)
            ang = manual_rotation_angle
        else:
            best_bgr, ang = find_best_orthogonal_orientation(img_bgr, is_first_page=(idx == 0))
        best_angles.append(ang)
        pil_oriented = Image.fromarray(cv2.cvtColor(best_bgr, cv2.COLOR_BGR2RGB))
        oriented_images.append(pil_oriented)

    first_page_img = oriented_images[0]
    p1_bgr = convert_to_opencv(first_page_img)
    from ocr.preprocessing import correct_document_perspective
    norm_p1_bgr, was_warped = correct_document_perspective(p1_bgr, 1600, 2200)

    t2_elapsed = time.time() - t2_start

    # STEP 3: STUDENT DETAILS OCR
    t3_start = time.time()
    if progress_box and progress_bar:
        progress_box.info("👤 Step 3/7: Extracting student details from ICFAI template ROIs...")
        progress_bar.progress(42)

    student_details, telemetry = extract_student_details_page1(first_page_img, detected_angle=best_angles[0])
    raw_p1_text = telemetry.get("raw_text", "")
    p1_confidence = telemetry.get("average_confidence", 85.0)
    t3_elapsed = time.time() - t3_start

    # STEP 4: MARKS TABLE GRID EXTRACTION
    t4_grid_start = time.time()
    if progress_box and progress_bar:
        progress_box.info("📊 Step 4/7: Extracting examiner marks table (Q1..Q10 grid)...")
        progress_bar.progress(57)

    marks_grid_results, grid_total_marks, grid_max_marks, table_overlay_img = extract_marks_table(norm_p1_bgr)
    t4_grid_elapsed = time.time() - t4_grid_start

    # Retrieve configured OCR Engine from Database
    sys_settings = db.get_settings()
    configured_engine = getattr(sys_settings, "ocr_engine", "paddleocr") or "paddleocr"

    # STEP 5: ANSWER EXTRACTION
    t5_ans_start = time.time()
    if progress_box and progress_bar:
        progress_box.info(f"📝 Step 5/7: Extracting student answers using [{configured_engine.upper()}] engine...")
        progress_bar.progress(71)

    answer_pages_oriented = oriented_images[1:] if len(oriented_images) > 1 else [oriented_images[0]]
    q_answers, raw_ans_text = extract_answers_from_pages(answer_pages_oriented, engine_name=configured_engine)
    t5_ans_elapsed = time.time() - t5_ans_start


    # STEP 6: EVALUATION & GRADINGS
    t6_eval_start = time.time()
    if progress_box and progress_bar:
        progress_box.info("🤖 Step 6/7: Evaluating answers & computing total marks/grade...")
        progress_bar.progress(85)

    subj_code = student_details.subject_code or "CS501"
    model_items = db.get_model_answers(subj_code)
    if not model_items:
        seed_sample_model_answers(db)
        model_items = db.get_model_answers("CS501")

    db.create_booklet_record(
        student=student_details,
        filename=uploaded_file.name,
        page_count=len(images),
        extracted_answers=q_answers
    )

    sys_settings = db.get_settings()
    eval_rec = evaluate_student_booklet(
        student_details=student_details,
        student_answers=q_answers,
        model_answer_items=model_items,
        booklet_filename=uploaded_file.name,
        settings=sys_settings
    )

    if grid_max_marks > 0:
        eval_rec.total_max_marks = grid_max_marks
    if grid_total_marks > 0:
        valid_final = min(eval_rec.total_max_marks, float(grid_total_marks))
        eval_rec.final_total_marks = valid_final
        eval_rec.percentage = min(100.0, round((valid_final / eval_rec.total_max_marks * 100.0), 2)) if eval_rec.total_max_marks > 0 else 0.0
        eval_rec.grade, _ = determine_grade_and_status(eval_rec.percentage, sys_settings)

    t6_eval_elapsed = time.time() - t6_eval_start

    # STEP 7: SAVE TO DATABASE
    t7_save_start = time.time()
    if progress_box and progress_bar:
        progress_box.info("💾 Step 7/7: Saving evaluation record & generating grade sheet...")
        progress_bar.progress(100)

    eval_id = db.save_evaluation(eval_rec)
    t7_save_elapsed = time.time() - t7_save_start

    raw_debug_text = raw_p1_text
    if len(images) > 1 and raw_ans_text:
        raw_debug_text += "\n\n=== ANSWER PAGES RAW OCR ===\n" + raw_ans_text

    total_pipeline_time = time.time() - pipeline_t0

    step_timings = {
        "loading_time": round(t1_elapsed, 2),
        "orientation_time": round(t2_elapsed, 2),
        "student_ocr_time": round(t3_elapsed, 2),
        "marks_table_time": round(t4_grid_elapsed, 2),
        "answer_ocr_time": round(t5_ans_elapsed, 2),
        "evaluation_time": round(t6_eval_elapsed, 2),
        "save_time": round(t7_save_elapsed, 2),
        "total_time": round(total_pipeline_time, 2)
    }

    result_dict = {
        "filename": uploaded_file.name,
        "student_details": student_details,
        "question_results": eval_rec.questions,
        "evaluation_record": eval_rec,
        "marks_grid_results": marks_grid_results,
        "table_overlay_img": table_overlay_img,
        "raw_ocr_debug_text": raw_debug_text,
        "oriented_images": oriented_images,
        "detected_angle": best_angles[0],
        "p1_confidence": p1_confidence,
        "page_count": len(images),
        "step_timings": step_timings
    }

    return result_dict
