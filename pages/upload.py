import streamlit as st
import time
import os
import cv2
import json
import re
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
from PIL import Image
from database.database import EvaluationDatabase
from models.database_models import StudentDetails
from utils.pdf_processor import convert_bytes_to_images, save_uploaded_file
from ocr.preprocessing import convert_to_opencv, find_best_orthogonal_orientation, correct_document_perspective
from ocr.orientation import rotate_image_angle
from ocr.first_page_ocr import process_first_page
from ocr.answer_ocr import extract_answers_from_pages
from ocr.marks_table import extract_marks_table
from evaluation.evaluator import evaluate_student_booklet
from evaluation.grading import determine_grade_and_status
from utils.sample_generator import seed_sample_model_answers
from template_config import (
    NORMALIZED_WIDTH,
    NORMALIZED_HEIGHT,
    STATUS_OK,
    STATUS_LOW_CONFIDENCE,
    STATUS_NEEDS_REVIEW,
    STATUS_NOT_DETECTED,
    STATUS_NOT_PRESENT
)
from utils.metrics import (
    COVER_FIELDS,
    compare_field_values,
    calculate_cover_field_metrics,
    get_student_field_value
)
from ocr.auto_verifier import (
    perform_automatic_verification,
    is_missing_value
)

def render_html(html_str: str):
    """
    Safely render HTML without markdown code-block conversion.
    Strips leading whitespace from each line so Markdown never treats 4-space indented HTML as code blocks.
    """
    clean_lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    st.markdown("\n".join(clean_lines), unsafe_allow_html=True)

def process_uploaded_booklet(
    uploaded_file,
    db: EvaluationDatabase,
    progress_box=None,
    progress_bar=None,
    manual_rotation_angle: Optional[int] = None
) -> dict:
    """
    MASTER PIPELINE PROCESSING FUNCTION:
    Executes sequential document analysis (Step 1 through Step 7) on the fixed booklet template.
    """
    print("========== PIPELINE DEBUG ==========")
    pipeline_t0 = time.time()

    # -------------------------------------------------------------
    # STEP 1: UPLOAD & VALIDATE DOCUMENT
    # -------------------------------------------------------------
    t1_start = time.time()
    print("[PIPELINE] STEP 1 START - Read & Validate Booklet")
    if progress_box and progress_bar:
        progress_box.info("📖 Step 1/7: Reading & validating booklet file...")
        progress_bar.progress(14)

    try:
        if hasattr(uploaded_file, "getvalue"):
            file_bytes = uploaded_file.getvalue()
        elif hasattr(uploaded_file, "read"):
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
        elif isinstance(uploaded_file, bytes):
            file_bytes = uploaded_file
        else:
            raise ValueError(f"Unsupported file type: {type(uploaded_file)}")

        fname = getattr(uploaded_file, "name", "uploaded_booklet.pdf")
        saved_path, unique_name = save_uploaded_file(file_bytes, fname)
        images = convert_bytes_to_images(file_bytes, fname)
        t1_elapsed = time.time() - t1_start
        print(f"[PERF] Step 1 END - {t1_elapsed:.2f} sec (Pages loaded: {len(images)})")
    except Exception as e:
        print(f"[STEP 1 ERROR] {repr(e)}")
        raise RuntimeError(f"Step 1 Reading Booklet failed: {e}")

    # -------------------------------------------------------------
    # STEP 2: ORIENTATION & PAGE NORMALIZATION
    # -------------------------------------------------------------
    t2_start = time.time()
    print("[PIPELINE] STEP 2 START - Orientation & Normalization")
    if progress_box and progress_bar:
        progress_box.info("🔄 Step 2/7: Orienting & normalizing pages (90° / 0° detection)...")
        progress_bar.progress(28)

    try:
        oriented_images = []
        best_angles = []
        for idx, img in enumerate(images):
            img_bgr = convert_to_opencv(img)
            if manual_rotation_angle is not None and manual_rotation_angle in [0, 90, 180, 270]:
                from ocr.orientation import rotate_image_angle
                rot_bgr = rotate_image_angle(img_bgr, manual_rotation_angle)
                rot_angle = manual_rotation_angle
            else:
                rot_bgr, rot_angle = find_best_orthogonal_orientation(img_bgr, is_first_page=(idx == 0))
            rot_pil = Image.fromarray(cv2.cvtColor(rot_bgr, cv2.COLOR_BGR2RGB))
            oriented_images.append(rot_pil)
            best_angles.append(rot_angle)
        
        # Prepare normalized 1600 x 2200 canvas copy of Page 1 for Step 3 ROI processing ONCE
        p1_bgr = convert_to_opencv(oriented_images[0])
        from ocr.preprocessing import correct_document_perspective
        norm_p1_bgr, was_warped = correct_document_perspective(p1_bgr, NORMALIZED_WIDTH, NORMALIZED_HEIGHT)
        norm_p1_pil = Image.fromarray(cv2.cvtColor(norm_p1_bgr, cv2.COLOR_BGR2RGB))

        t2_elapsed = time.time() - t2_start
        print(f"[PERF] Step 2 END - {t2_elapsed:.2f} sec (Orientation: {best_angles[0]}°)")

    except Exception as e:
        print(f"[STEP 2 ERROR] {repr(e)}")
        raise RuntimeError(f"Step 2 Page Orientation failed: {e}")

    # -------------------------------------------------------------
    # STEP 3: FIXED TEMPLATE FIELD EXTRACTION (STUDENT DETAILS)
    # -------------------------------------------------------------
    t3_start = time.time()
    print("[PIPELINE] STEP 3 START - Extracting Fixed Template Fields")
    if progress_box and progress_bar:
        progress_box.info("🔍 Step 3/7: Reading 22 student & examination template fields...")
        progress_bar.progress(42)

    try:
        student_details, raw_p1_text, p1_confidence = process_first_page(norm_p1_pil, detected_angle=best_angles[0])
        setattr(student_details, "detected_angle", best_angles[0])
        t3_elapsed = time.time() - t3_start
        print(f"[STEP 3] Student details extracted: {student_details.student_name or 'N/A'}")
        print(f"[PERF] Step 3 END - {t3_elapsed:.2f} sec")
    except Exception as e:
        print(f"[STEP 3 ERROR] {repr(e)}")
        raise RuntimeError(f"Step 3 Reading Student Details failed: {e}")

    # -------------------------------------------------------------
    # STEP 4: EXAMINER MARKS TABLE EXTRACTION
    # -------------------------------------------------------------
    t4_grid_start = time.time()
    print("[PIPELINE] STEP 4 START - Marks Table Grid Extraction")
    if progress_box and progress_bar:
        progress_box.info("📊 Step 4/7: Extracting examiner marks table (Q1..Q10 grid)...")
        progress_bar.progress(57)

    try:
        marks_grid_results, grid_total_marks, grid_max_marks, table_overlay_img = extract_marks_table(norm_p1_bgr)
        t4_grid_elapsed = time.time() - t4_grid_start
        print(f"[STEP 4] Examiner marks table extracted: Grand Total = {grid_total_marks}")
        print(f"[PERF] Step 4 END - {t4_grid_elapsed:.2f} sec")
    except Exception as e:
        print(f"[STEP 4 ERROR] {repr(e)}")
        raise RuntimeError(f"Step 4 Marks Table Extraction failed: {e}")

    # -------------------------------------------------------------
    # STEP 5: ANSWER PAGE OCR EXTRACTION
    # -------------------------------------------------------------
    t5_ans_start = time.time()
    print("[PIPELINE] STEP 5 START - Answer Extraction")
    if progress_box and progress_bar:
        progress_box.info("📝 Step 5/7: Extracting handwritten student answers...")
        progress_bar.progress(71)

    try:
        answer_pages_oriented = oriented_images[1:] if len(oriented_images) > 1 else [oriented_images[0]]
        q_answers, raw_ans_text = extract_answers_from_pages(answer_pages_oriented)
        t5_ans_elapsed = time.time() - t5_ans_start
        print(f"[STEP 5] Answers extracted: {list(q_answers.keys())}")
        print(f"[PERF] Step 5 END - {t5_ans_elapsed:.2f} sec")
    except Exception as e:
        print(f"[STEP 5 ERROR] {repr(e)}")
        raise RuntimeError(f"Step 5 Reading Answers failed: {e}")

    # -------------------------------------------------------------
    # STEP 6: AI MARKS EVALUATION & GRADE CALCULATION
    # -------------------------------------------------------------
    t6_eval_start = time.time()
    print("[PIPELINE] STEP 6 START - AI Evaluation")
    if progress_box and progress_bar:
        progress_box.info("🤖 Step 6/7: Evaluating answers & computing total marks/grade...")
        progress_bar.progress(85)

    try:
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

        # Update total score with examiner written marks table total
        if grid_max_marks > 0:
            eval_rec.total_max_marks = grid_max_marks
        if grid_total_marks > 0:
            valid_final = min(eval_rec.total_max_marks, float(grid_total_marks))
            eval_rec.final_total_marks = valid_final
            eval_rec.percentage = min(100.0, round((valid_final / eval_rec.total_max_marks * 100.0), 2)) if eval_rec.total_max_marks > 0 else 0.0
            eval_rec.grade, _ = determine_grade_and_status(eval_rec.percentage, sys_settings)

        t6_eval_elapsed = time.time() - t6_eval_start
        print(f"[PERF] Step 6 END - {t6_eval_elapsed:.2f} sec")
    except Exception as e:
        print(f"[STEP 6 ERROR] {repr(e)}")
        raise RuntimeError(f"Step 6 Evaluating Answers failed: {e}")

    # -------------------------------------------------------------
    # STEP 7: PREPARE EVALUATION RECORD (PENDING VERIFICATION)
    # -------------------------------------------------------------
    t7_save_start = time.time()
    print("[PIPELINE] STEP 7 START - Preparing Evaluation Record")
    if progress_box and progress_bar:
        progress_box.info("💾 Step 7/7: Preparing evaluation summary (pending manual verification)...")
        progress_bar.progress(100)

    # Do NOT automatically commit unverified OCR results as confirmed data
    eval_rec.is_approved = False
    eval_id = None
    t7_save_elapsed = time.time() - t7_save_start
    print(f"[PERF] Step 7 END - {t7_save_elapsed:.2f} sec (Prepared draft record)")
    print("[PIPELINE] ALL 7 PIPELINE STEPS COMPLETED")

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
        "prep_time": round(t7_save_elapsed, 2),
        "total_time": round(total_pipeline_time, 2)
    }

    print("\n==================================================")
    print("[PERFORMANCE SUMMARY - 7 STEPS]")
    print(f"Step 1 Reading Booklet:         {t1_elapsed:.2f} sec")
    print(f"Step 2 Page Orientation:        {t2_elapsed:.2f} sec")
    print(f"Step 3 Student Details OCR:     {t3_elapsed:.2f} sec")
    print(f"Step 4 Marks Table Grid OCR:    {t4_grid_elapsed:.2f} sec")
    print(f"Step 5 Answer Extraction:       {t5_ans_elapsed:.2f} sec")
    print(f"Step 6 Evaluation & Grading:    {t6_eval_elapsed:.2f} sec")
    print(f"Step 7 Prep & Draft:            {t7_save_elapsed:.2f} sec")
    print(f"Total 7-Step Pipeline Time:     {total_pipeline_time:.2f} sec")
    print("==================================================\n")

    clean_fn = re.sub(r'[^a-zA-Z0-9_-]', '_', uploaded_file.name)
    initial_fields = {
        f["id"]: get_student_field_value(student_details, f["id"]) for f in COVER_FIELDS
    }
    result_dict = {
        "student_index": 0,
        "student_uid": f"{clean_fn}_s1",
        "filename": uploaded_file.name,
        "page_num": 1,
        "total_students": 1,
        "student_name": student_details.student_name or "Student 1",
        "enrollment_no": getattr(student_details, "roll_number", "") or getattr(student_details, "registration_number", "") or "",
        "student_details": student_details,
        "question_results": eval_rec.questions,
        "evaluation_record": eval_rec,
        "marks_grid_results": marks_grid_results,
        "table_overlay_img": table_overlay_img,
        "raw_ocr_debug_text": raw_debug_text,
        "original_page_img": images[0] if images else None,
        "oriented_images": oriented_images,
        "detected_angle": best_angles[0],
        "p1_confidence": p1_confidence,
        "page_count": len(images),
        "step_timings": step_timings,
        "is_saved": False,
        "saved_eval_id": None,
        "initial_extracted_fields": initial_fields
    }

    return result_dict


def process_single_page_student(
    page_img: Image.Image,
    student_index: int,
    total_students: int,
    filename: str,
    page_num: int,
    db: EvaluationDatabase,
    manual_rotation_angle: Optional[int] = None
) -> dict:
    """
    Process an individual student page from a multi-student PDF or image batch.
    Executes orientation detection, template extraction (all 11 fields), marks table OCR,
    and returns a self-contained student record dictionary.
    """
    t0 = time.time()
    img_bgr = convert_to_opencv(page_img)

    # Orientation detection & rotation
    t_ori_0 = time.time()
    if manual_rotation_angle is not None and manual_rotation_angle in [0, 90, 180, 270]:
        rot_bgr = rotate_image_angle(img_bgr, manual_rotation_angle)
        rot_angle = manual_rotation_angle
    else:
        rot_bgr, rot_angle = find_best_orthogonal_orientation(img_bgr, is_first_page=True)
    rot_pil = Image.fromarray(cv2.cvtColor(rot_bgr, cv2.COLOR_BGR2RGB))
    t_ori = time.time() - t_ori_0

    # Perspective correction & normalization to 1600x2200 canvas copy
    norm_p1_bgr, was_warped = correct_document_perspective(rot_bgr, NORMALIZED_WIDTH, NORMALIZED_HEIGHT)
    norm_p1_pil = Image.fromarray(cv2.cvtColor(norm_p1_bgr, cv2.COLOR_BGR2RGB))

    # Fixed template field extraction (11 fields)
    t_ocr_0 = time.time()
    student_details, raw_p1_text, p1_confidence = process_first_page(norm_p1_pil, detected_angle=rot_angle)
    setattr(student_details, "detected_angle", rot_angle)
    t_ocr = time.time() - t_ocr_0

    # Examiner marks table extraction
    t_grid_0 = time.time()
    try:
        marks_grid_results, grid_total_marks, grid_max_marks, table_overlay_img = extract_marks_table(norm_p1_bgr)
    except Exception as e:
        print(f"[MARKS GRID WARN] {e}")
        marks_grid_results, grid_total_marks, grid_max_marks, table_overlay_img = [], 0.0, 0, None
    t_grid = time.time() - t_grid_0

    # Answer page OCR extraction (from this student sheet)
    t_ans_0 = time.time()
    try:
        q_answers, raw_ans_text = extract_answers_from_pages([rot_pil])
    except Exception as e:
        print(f"[ANS WARN] {e}")
        q_answers, raw_ans_text = {}, ""
    t_ans = time.time() - t_ans_0

    # AI evaluation & grade computation
    t_eval_0 = time.time()
    subj_code = student_details.subject_code or "CS501"
    model_items = db.get_model_answers(subj_code)
    if not model_items:
        seed_sample_model_answers(db)
        model_items = db.get_model_answers("CS501")

    sys_settings = db.get_settings()
    eval_rec = evaluate_student_booklet(
        student_details=student_details,
        student_answers=q_answers,
        model_answer_items=model_items,
        booklet_filename=f"{filename}_Page_{page_num}",
        settings=sys_settings
    )

    if grid_max_marks > 0:
        eval_rec.total_max_marks = grid_max_marks
    if grid_total_marks > 0:
        valid_final = min(eval_rec.total_max_marks, float(grid_total_marks))
        eval_rec.final_total_marks = valid_final
        eval_rec.percentage = min(100.0, round((valid_final / eval_rec.total_max_marks * 100.0), 2)) if eval_rec.total_max_marks > 0 else 0.0
        eval_rec.grade, _ = determine_grade_and_status(eval_rec.percentage, sys_settings)

    eval_rec.is_approved = False
    t_eval = time.time() - t_eval_0

    # Extract initial field values for all 11 fields
    initial_fields = {
        f["id"]: get_student_field_value(student_details, f["id"]) for f in COVER_FIELDS
    }

    clean_fn = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
    student_uid = f"{clean_fn}_p{page_num}_s{student_index + 1}"

    student_name = student_details.student_name or f"Student {student_index + 1}"
    enrollment_no = (
        getattr(student_details, "roll_number", "") or
        getattr(student_details, "registration_number", "") or
        ""
    )

    total_pipeline_time = time.time() - t0
    step_timings = {
        "orientation_time": round(t_ori, 2),
        "student_ocr_time": round(t_ocr, 2),
        "marks_table_time": round(t_grid, 2),
        "answer_ocr_time": round(t_ans, 2),
        "evaluation_time": round(t_eval, 2),
        "total_time": round(total_pipeline_time, 2)
    }

    raw_debug_text = raw_p1_text
    if raw_ans_text:
        raw_debug_text += "\n\n=== ANSWER PAGES RAW OCR ===\n" + raw_ans_text

    return {
        "student_index": student_index,
        "student_uid": student_uid,
        "filename": filename,
        "page_num": page_num,
        "total_students": total_students,
        "student_name": student_name,
        "enrollment_no": enrollment_no,
        "student_details": student_details,
        "question_results": eval_rec.questions,
        "evaluation_record": eval_rec,
        "marks_grid_results": marks_grid_results,
        "table_overlay_img": table_overlay_img,
        "raw_ocr_debug_text": raw_debug_text,
        "original_page_img": page_img,
        "oriented_images": [rot_pil],
        "detected_angle": rot_angle,
        "p1_confidence": p1_confidence,
        "page_count": 1,
        "step_timings": step_timings,
        "is_saved": False,
        "saved_eval_id": None,
        "initial_extracted_fields": initial_fields
    }


def process_multi_student_booklet_files(
    uploaded_files: list,
    db: EvaluationDatabase,
    progress_box=None,
    progress_bar=None,
    is_multi_student_mode: bool = True
) -> List[dict]:
    """
    Reads uploaded files (PDFs / Images).
    In multi_student_mode, extracts each page as a separate student record.
    In single_student_mode, treats first page as details and subsequent pages as answers for 1 student.
    """
    all_pages = []
    for f in uploaded_files:
        if hasattr(f, "getvalue"):
            f_bytes = f.getvalue()
        elif hasattr(f, "read"):
            if hasattr(f, "seek"):
                f.seek(0)
            f_bytes = f.read()
        elif isinstance(f, bytes):
            f_bytes = f
        else:
            continue
        fname = getattr(f, "name", "uploaded_file.pdf")
        # Save file to uploads folder
        try:
            save_uploaded_file(f_bytes, fname)
        except Exception:
            pass
        imgs = convert_bytes_to_images(f_bytes, fname)
        for pnum, img in enumerate(imgs, 1):
            all_pages.append((fname, pnum, img))

    total_records = len(all_pages)
    if total_records == 0:
        raise ValueError("No valid student answer pages found in the uploaded file(s).")

    # If single booklet mode selected and more than 1 page from 1 file:
    if not is_multi_student_mode and len(uploaded_files) == 1 and total_records > 1:
        single_result = process_uploaded_booklet(
            uploaded_files[0],
            db,
            progress_box=progress_box,
            progress_bar=progress_bar
        )
        return [single_result]

    # Multi-student mode: process each page as a separate student record
    records = []
    for idx, (fname, pnum, page_img) in enumerate(all_pages):
        if progress_box and progress_bar:
            pct = int((idx / total_records) * 100)
            progress_bar.progress(pct)
            progress_box.info(f"⏳ Processing Student {idx + 1} of {total_records} (Page {pnum} from {fname})...")

        try:
            student_rec = process_single_page_student(
                page_img=page_img,
                student_index=idx,
                total_students=total_records,
                filename=fname,
                page_num=pnum,
                db=db
            )
            records.append(student_rec)
        except Exception as e:
            print(f"[STUDENT {idx+1} EXTRACTION FAILED] {e}")
            clean_fn = re.sub(r'[^a-zA-Z0-9_-]', '_', fname)
            fallback_details = StudentDetails(student_name=f"Student {idx + 1} (Unreadable)")
            from models.database_models import EvaluationRecord
            fallback_eval = EvaluationRecord(
                student=fallback_details,
                booklet_filename=fname,
                page_count=1,
                total_max_marks=100.0,
                final_total_marks=0.0,
                percentage=0.0,
                grade="F",
                status="Fail",
                questions=[]
            )
            records.append({
                "student_index": idx,
                "student_uid": f"{clean_fn}_p{pnum}_s{idx + 1}",
                "filename": fname,
                "page_num": pnum,
                "total_students": total_records,
                "student_name": f"Student {idx + 1} (Unreadable)",
                "enrollment_no": "",
                "student_details": fallback_details,
                "question_results": [],
                "evaluation_record": fallback_eval,
                "marks_grid_results": [],
                "table_overlay_img": None,
                "raw_ocr_debug_text": f"OCR Extraction Failed on Page {pnum}: {e}",
                "original_page_img": page_img,
                "oriented_images": [page_img],
                "detected_angle": 0,
                "p1_confidence": 0.0,
                "page_count": 1,
                "step_timings": {},
                "is_saved": False,
                "saved_eval_id": None,
                "initial_extracted_fields": {f["id"]: "" for f in COVER_FIELDS}
            })

    if progress_box and progress_bar:
        progress_bar.progress(100)

    return records


def render_student_navigation(student_records: List[dict]):
    """
    Renders top navigation bar:
    - Previous and Next buttons (disabled at boundaries)
    - Student indicator counter (e.g., 'Student 1 of 6')
    - Direct jump dropdown selector with safe dynamic key
    """
    total = len(student_records)
    if total <= 1:
        return

    cur_idx = st.session_state.get("current_student_index", 0)
    cur_idx = max(0, min(cur_idx, total - 1))
    st.session_state["current_student_index"] = cur_idx
    cur_rec = student_records[cur_idx]

    c_prev, c_center, c_drop, c_next = st.columns([1.3, 2.5, 2.4, 1.3])

    with c_prev:
        is_prev_disabled = (cur_idx <= 0)
        if st.button("◀ Previous", key="btn_nav_prev", disabled=is_prev_disabled, width="stretch", help="View previous student record"):
            new_idx = max(0, cur_idx - 1)
            st.session_state["current_student_index"] = new_idx
            st.rerun()

    with c_center:
        st_name = cur_rec.get("student_name") or getattr(cur_rec.get("student_details"), "student_name", "") or f"Student {cur_idx + 1}"
        roll = cur_rec.get("enrollment_no") or getattr(cur_rec.get("student_details"), "roll_number", "") or "No Roll"
        render_html(f"""
        <div style="text-align: center; font-family: 'Inter', -apple-system, sans-serif; padding: 2px 0;">
            <div style="font-size: 15px; font-weight: 800; color: #F8FAFC; letter-spacing: 0.2px;">
                Student <span style="color: #38BDF8;">{cur_idx + 1}</span> of <span style="color: #94A3B8;">{total}</span>
            </div>
            <div style="font-size: 12px; color: #94A3B8; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                <b style="color: #E2E8F0;">{st_name}</b> • <span style="font-family: monospace;">{roll}</span> (Page {cur_rec.get('page_num', cur_idx + 1)})
            </div>
        </div>
        """)

    with c_drop:
        student_labels = []
        for i, rec in enumerate(student_records):
            sname = rec.get("student_name") or getattr(rec.get("student_details"), "student_name", "") or f"Student {i+1}"
            sroll = rec.get("enrollment_no") or getattr(rec.get("student_details"), "roll_number", "") or ""
            sroll_str = f" - {sroll}" if sroll else ""
            student_labels.append(f"Student {i+1} of {total}: {sname}{sroll_str}")

        selected_idx = st.selectbox(
            "Jump to Student",
            options=list(range(total)),
            format_func=lambda i: student_labels[i] if 0 <= i < len(student_labels) else f"Student {i+1}",
            index=cur_idx,
            key=f"jump_student_select_{cur_idx}",
            label_visibility="collapsed"
        )
        if selected_idx != cur_idx:
            st.session_state["current_student_index"] = selected_idx
            st.rerun()

    with c_next:
        is_next_disabled = (cur_idx >= total - 1)
        if st.button("Next ▶", key="btn_nav_next", disabled=is_next_disabled, width="stretch", help="View next student record"):
            new_idx = min(total - 1, cur_idx + 1)
            st.session_state["current_student_index"] = new_idx
            st.rerun()

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)



def render_student_result(result: dict, db: EvaluationDatabase, student_records: Optional[List[dict]] = None, cur_idx: int = 0):
    """Renders persistent grade sheet, STUDENT RESULT SUMMARY card, and teacher override workspace for an individual student."""
    eval_rec = result["evaluation_record"]
    student_info = result["student_details"]
    oriented_images = result.get("oriented_images", [])
    first_page_img = oriented_images[0] if oriented_images else None
    answer_page_imgs = oriented_images[1:] if len(oriented_images) > 1 else oriented_images
    sys_settings = db.get_settings()
    telemetry = getattr(student_info, "telemetry", {})
    marks_grid_results = result.get("marks_grid_results", [])
    table_overlay_img = result.get("table_overlay_img")
    step_timings = result.get("step_timings", {})
    crops_dict = telemetry.get("crops", {})

    # -------------------------------------------------------------
    # COVER-PAGE FIELDS GROUND-TRUTH VERIFICATION & EXTRACTION
    # -------------------------------------------------------------
    def get_extracted_model_val(fid):
        return get_student_field_value(student_info, fid)

    def is_uncertain(field_key, val):
        fdata = crops_dict.get(field_key, {})
        status = fdata.get("status", "")
        conf = fdata.get("confidence", 0.0)
        is_empty = not val or str(val).strip().lower() in [
            "not detected", "none", "not_detected", "needs review", "not present", "not_present", ""
        ]
        return (
            is_empty or 
            status in [STATUS_LOW_CONFIDENCE, STATUS_NEEDS_REVIEW, STATUS_NOT_DETECTED, STATUS_NOT_PRESENT] or 
            conf < 65.0
        )

    # Preserve initial OCR output snapshot
    if "initial_extracted_fields" not in result:
        result["initial_extracted_fields"] = {
            f["id"]: get_extracted_model_val(f["id"]) for f in COVER_FIELDS
        }

    student_uid = str(result.get("student_uid", f"student_{cur_idx + 1}"))
    ver_state_key = f"ver_state_{student_uid}"

    needs_init = (ver_state_key not in st.session_state)
    raw_roll = str(result["initial_extracted_fields"].get("enrollment_no", ""))
    clean_roll = re.sub(r"[^A-Za-z0-9]", "", raw_roll).upper()

    if needs_init:
        # Check SQLite persistent storage first
        saved_db_data = db.get_field_verifications(student_uid, raw_roll)
        if saved_db_data and saved_db_data.get("field_data"):
            st.session_state[ver_state_key] = saved_db_data["field_data"]
        else:
            # Check if trusted reference data / ground truth was supplied
            trusted_roster = st.session_state.get("trusted_reference_data", {})
            matching_ref = (
                trusted_roster.get(clean_roll) or
                trusted_roster.get(raw_roll) or
                trusted_roster.get(f"student_{cur_idx + 1}") or
                {}
            )

            # Perform automatic verification against original PDF crops & rules
            auto_v_map = perform_automatic_verification(
                initial_extracted_fields=result["initial_extracted_fields"],
                crops_dict=crops_dict,
                cover_fields=COVER_FIELDS,
                trusted_reference=matching_ref if matching_ref else None
            )
            st.session_state[ver_state_key] = auto_v_map

            # Automatically persist initial verification state to SQLite
            try:
                init_metrics = calculate_cover_field_metrics(
                    COVER_FIELDS,
                    auto_v_map,
                    result["initial_extracted_fields"]
                )
                db.save_field_verifications(
                    student_uid=student_uid,
                    booklet_filename=result.get("filename", "booklet"),
                    v_map=auto_v_map,
                    metrics=init_metrics,
                    roll_number=raw_roll
                )
            except Exception:
                pass

    # Synchronize with any active widget text inputs from session_state
    v_map = st.session_state[ver_state_key]
    for f in COVER_FIELDS:
        fid = f["id"]
        w_text_key = f"edit_gt_{fid}_{student_uid}"
        if w_text_key in st.session_state and st.session_state[w_text_key].strip():
            v_map[fid]["ground_truth"] = st.session_state[w_text_key].strip()

    # Dynamic accuracy & error calculation for all 11 fields for this student
    metrics = calculate_cover_field_metrics(
        COVER_FIELDS,
        v_map,
        result["initial_extracted_fields"]
    )
    total_fields = metrics["total_fields"]
    verified_correct_count = metrics["verified_correct_count"]
    incorrect_count = metrics["incorrect_count"]
    needs_verification_count = metrics["needs_verification_count"]
    field_status_map = metrics["field_status_map"]
    accuracy_pct = metrics["accuracy_pct"]
    error_pct = metrics["error_pct"]
    pending_pct = metrics["pending_pct"]
    accuracy_str = metrics["accuracy_str"]
    error_str = metrics["error_str"]
    pending_str = metrics["pending_str"]
    accuracy_sub = metrics["accuracy_sub"]
    error_sub = metrics["error_sub"]
    pending_sub = metrics["pending_sub"]

    unverified_fields = [f["label"] for f in COVER_FIELDS if field_status_map.get(f["id"]) == "Needs Verification"]

    # Status header and reset actions
    c_hdr1, c_hdr2 = st.columns([2.0, 2.0])
    with c_hdr1:
        if result.get("is_saved"):
            st.success(f"🟢 **Status: Confirmed & Saved** in SQLite database (Record #{result.get('saved_eval_id')})")
        else:
            st.info("🟡 **Status: Extracted Draft** — Review fields or click Confirm & Save to store.")
    with c_hdr2:
        c_rst1, c_rst2, c_rst3 = st.columns([1.2, 1.2, 1])
        with c_rst1:
            if st.button("✓ Verify All", key=f"btn_verify_all_{student_uid}", type="primary", width="stretch", help="Mark all detected fields as verified for this student"):
                for f in COVER_FIELDS:
                    fid = f["id"]
                    ocr_val = result["initial_extracted_fields"].get(fid, "")
                    is_missing = not ocr_val or str(ocr_val).strip().lower() in ["not detected", "not_detected", "none", "not present", "needs review", ""]
                    val_to_use = ocr_val if not is_missing else (v_map[fid].get("ground_truth", "") or "")
                    if val_to_use:
                        v_map[fid]["status"] = "Verified"
                        v_map[fid]["verified"] = True
                        v_map[fid]["is_explicit_incorrect"] = False
                        v_map[fid]["ground_truth"] = val_to_use
                        st.session_state[f"edit_gt_{fid}_{student_uid}"] = val_to_use
                try:
                    up_metrics = calculate_cover_field_metrics(COVER_FIELDS, v_map, result["initial_extracted_fields"])
                    db.save_field_verifications(student_uid, result.get("filename", "booklet"), v_map, up_metrics, raw_roll)
                except Exception:
                    pass
                st.rerun()
        with c_rst2:
            if st.button("🔄 Auto-Verify", key=f"btn_reset_curr_{student_uid}", type="secondary", width="stretch", help="Re-run automatic verification against original PDF"):
                trusted_roster = st.session_state.get("trusted_reference_data", {})
                matching_ref = (
                    trusted_roster.get(clean_roll) or
                    trusted_roster.get(raw_roll) or
                    trusted_roster.get(f"student_{cur_idx + 1}") or
                    {}
                )
                auto_v_map = perform_automatic_verification(
                    initial_extracted_fields=result["initial_extracted_fields"],
                    crops_dict=crops_dict,
                    cover_fields=COVER_FIELDS,
                    trusted_reference=matching_ref if matching_ref else None
                )
                st.session_state[ver_state_key] = auto_v_map
                for f in COVER_FIELDS:
                    st.session_state.pop(f"edit_gt_{f['id']}_{student_uid}", None)
                try:
                    up_metrics = calculate_cover_field_metrics(COVER_FIELDS, auto_v_map, result["initial_extracted_fields"])
                    db.save_field_verifications(student_uid, result.get("filename", "booklet"), auto_v_map, up_metrics, raw_roll)
                except Exception:
                    pass
                st.rerun()
        with c_rst3:
            if st.button("🗑️ Clear Batch", key=f"btn_clear_all_{student_uid}", type="secondary", width="stretch", help="Clear all uploaded student records"):
                st.session_state["student_records"] = []
                st.session_state["current_student_index"] = 0
                st.session_state["processed_upload_key"] = None
                st.session_state["evaluation_result"] = None
                st.session_state["processed_file_id"] = None
                st.rerun()

    # Honest confidence notification: never display misleading overall high confidence if any key field is unreliable
    if unverified_fields:
        st.warning(f"⚠️ **Needs Verification ({len(unverified_fields)} fields):** `{', '.join(unverified_fields)}` require manual review before saving.")
    else:
        avg_conf = telemetry.get("average_confidence", result.get("p1_confidence", 85.0))
        st.success(f"✓ **High Confidence:** All key examination metadata extracted reliably (Avg confidence: {avg_conf:.1f}%).")

    # -------------------------------------------------------------
    # MAIN WORKSPACE: PREVIEW & CLEAN TEXT SUMMARY
    # -------------------------------------------------------------
    col_preview, col_summary = st.columns([1, 1])

    with col_preview:
        st.write(f"##### 📄 Student Answer Sheet Preview (Page {result.get('page_num', cur_idx + 1)})")

        # Interactive Orientation Controls: allow manual adjustment and re-run for this specific student
        st.caption("🔄 **Page Orientation Controls:**")
        rot_c1, rot_c2, rot_c3, rot_c4 = st.columns([1, 1, 1, 1])
        with rot_c1:
            if st.button("↺ 90° CCW", key=f"btn_rot_ccw_{student_uid}", help="Rotate image 90° counter-clockwise", width="stretch"):
                cur_ang = st.session_state.get(f"rot_angle_{student_uid}", result.get("detected_angle", 0))
                new_ang = (cur_ang + 270) % 360
                st.session_state[f"rot_angle_{student_uid}"] = new_ang
                if result.get("original_page_img"):
                    with st.spinner("Re-processing student page with new orientation..."):
                        updated_rec = process_single_page_student(
                            page_img=result["original_page_img"],
                            student_index=result.get("student_index", cur_idx),
                            total_students=result.get("total_students", 1),
                            filename=result.get("filename", "booklet"),
                            page_num=result.get("page_num", 1),
                            db=db,
                            manual_rotation_angle=new_ang
                        )
                        if student_records is not None and cur_idx < len(student_records):
                            student_records[cur_idx] = updated_rec
                            st.session_state["student_records"] = student_records
                        st.session_state["evaluation_result"] = updated_rec
                        st.rerun()
        with rot_c2:
            if st.button("↻ 90° CW", key=f"btn_rot_cw_{student_uid}", help="Rotate image 90° clockwise", width="stretch"):
                cur_ang = st.session_state.get(f"rot_angle_{student_uid}", result.get("detected_angle", 0))
                new_ang = (cur_ang + 90) % 360
                st.session_state[f"rot_angle_{student_uid}"] = new_ang
                if result.get("original_page_img"):
                    with st.spinner("Re-processing student page with new orientation..."):
                        updated_rec = process_single_page_student(
                            page_img=result["original_page_img"],
                            student_index=result.get("student_index", cur_idx),
                            total_students=result.get("total_students", 1),
                            filename=result.get("filename", "booklet"),
                            page_num=result.get("page_num", 1),
                            db=db,
                            manual_rotation_angle=new_ang
                        )
                        if student_records is not None and cur_idx < len(student_records):
                            student_records[cur_idx] = updated_rec
                            st.session_state["student_records"] = student_records
                        st.session_state["evaluation_result"] = updated_rec
                        st.rerun()
        with rot_c3:
            if st.button("🔄 180°", key=f"btn_rot_180_{student_uid}", help="Flip image upside down (180°)", width="stretch"):
                cur_ang = st.session_state.get(f"rot_angle_{student_uid}", result.get("detected_angle", 0))
                new_ang = (cur_ang + 180) % 360
                st.session_state[f"rot_angle_{student_uid}"] = new_ang
                if result.get("original_page_img"):
                    with st.spinner("Re-processing student page with new orientation..."):
                        updated_rec = process_single_page_student(
                            page_img=result["original_page_img"],
                            student_index=result.get("student_index", cur_idx),
                            total_students=result.get("total_students", 1),
                            filename=result.get("filename", "booklet"),
                            page_num=result.get("page_num", 1),
                            db=db,
                            manual_rotation_angle=new_ang
                        )
                        if student_records is not None and cur_idx < len(student_records):
                            student_records[cur_idx] = updated_rec
                            st.session_state["student_records"] = student_records
                        st.session_state["evaluation_result"] = updated_rec
                        st.rerun()
        with rot_c4:
            if st.button("⚡ Re-run", key=f"btn_rerun_ocr_{student_uid}", help="Re-run evaluation on current orientation", width="stretch"):
                cur_ang = st.session_state.get(f"rot_angle_{student_uid}", result.get("detected_angle", 0))
                if result.get("original_page_img"):
                    with st.spinner("Re-running OCR on student page..."):
                        updated_rec = process_single_page_student(
                            page_img=result["original_page_img"],
                            student_index=result.get("student_index", cur_idx),
                            total_students=result.get("total_students", 1),
                            filename=result.get("filename", "booklet"),
                            page_num=result.get("page_num", 1),
                            db=db,
                            manual_rotation_angle=cur_ang
                        )
                        if student_records is not None and cur_idx < len(student_records):
                            student_records[cur_idx] = updated_rec
                            st.session_state["student_records"] = student_records
                        st.session_state["evaluation_result"] = updated_rec
                        st.rerun()

        if first_page_img:
            st.image(first_page_img, caption=f"Page {result.get('page_num', cur_idx + 1)}: Student Sheet (Orientation: {result.get('detected_angle', 0)}°)", width="stretch")
        else:
            st.info("No page image preview available.")

    with col_summary:
        st.write("##### 📋 Extracted Information Summary")

        def val_clean(v, fallback="Not Detected"):
            if v and str(v).strip() and str(v).strip().lower() not in ["not detected", "not_detected", "none", "needs review"]:
                return str(v).strip()
            return fallback

        summary_rows_html = []
        for f in COVER_FIELDS:
            fkey = f["id"]
            label = f["label"]
            field_state = v_map.get(fkey, {})
            is_ver = field_state.get("verified", False)
            ocr_val = result["initial_extracted_fields"].get(fkey, "")
            gt_val = field_state.get("ground_truth", "")

            raw_disp = gt_val if (is_ver and gt_val) else ocr_val
            disp = val_clean(raw_disp, "Not Detected")

            fdata = crops_dict.get(fkey, {})
            fconf = fdata.get("confidence", 0.0)
            is_detected = (disp != "Not Detected" and disp != "Not Present" and disp != "—")
            if is_detected and fconf > 0:
                conf_pill = f"<span style='font-size:11px; color:#475569; background:#F1F5F9; border:1px solid #CBD5E1; padding:2px 7px; border-radius:6px; font-weight:600; margin-left:6px;' title='Raw OCR Confidence'>{fconf:.0f}% Conf</span>"
            elif is_detected:
                conf_pill = "<span style='font-size:11px; color:#64748B; background:#F8FAFC; border:1px solid #E2E8F0; padding:2px 6px; border-radius:6px; margin-left:6px;'>Detected</span>"
            else:
                conf_pill = ""

            f_stat = field_status_map.get(fkey, "Needs Verification")
            if f_stat == "Needs Verification":
                badge = "<span style='color:#B45309; font-weight:600; font-size:12px; background:#FEF3C7; padding:2px 8px; border-radius:12px;'>⚠️ Needs Verification</span>"
            elif f_stat == "Verified Correct":
                badge = "<span style='color:#16A34A; font-weight:600; font-size:12px; background:#DCFCE7; padding:2px 8px; border-radius:12px;'>✓ Verified Correct</span>"
            else:
                badge = "<span style='color:#DC2626; font-weight:600; font-size:12px; background:#FEE2E2; padding:2px 8px; border-radius:12px;'>✗ Incorrect</span>"

            summary_rows_html.append(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid #F1F5F9;">
                <div>
                    <span style="color:#64748B; font-size:13px; font-weight:500;">{label}:</span> 
                    <b style="color:#0F172A; font-size:14px; margin-left:6px;">{disp}</b>
                    {conf_pill}
                </div>
                <div>{badge}</div>
            </div>
            """)

        all_rows_str = "".join(summary_rows_html)
        render_html(f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:18px 20px; box-shadow:0 2px 8px rgba(0,0,0,0.03);">
            {all_rows_str}
        </div>
        """)

        # -------------------------------------------------------------
        # THREE DYNAMIC METRIC CARDS: ACCURACY, ERROR, & PENDING VERIFICATION
        # -------------------------------------------------------------
        c_card1, c_card2, c_card3 = st.columns(3)
        with c_card1:
            render_html(f"""
            <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 14px; padding: 16px 14px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12); position: relative; overflow: hidden; min-height: 148px; display: flex; flex-direction: column; justify-content: space-between; margin-top: 14px; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;">
                <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #10B981, #059669);"></div>
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">ACCURACY (%)</span>
                        <div style="width: 28px; height: 28px; border-radius: 8px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.35); display: flex; align-items: center; justify-content: center;">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="10"></circle>
                                <path d="m9 12 2 2 4-4"></path>
                            </svg>
                        </div>
                    </div>
                    <div style="display: flex; align-items: baseline; margin-bottom: 10px;">
                        <span style="font-size: 26px; font-weight: 900; color: #10B981; letter-spacing: -0.5px; line-height: 1;">{accuracy_str}</span>
                    </div>
                </div>
                <div>
                    <div style="width: 100%; height: 7px; background: #1E293B; border-radius: 9999px; overflow: hidden; margin-bottom: 6px;">
                        <div style="width: {accuracy_pct}%; height: 100%; background: linear-gradient(90deg, #10B981, #34D399); border-radius: 9999px;"></div>
                    </div>
                    <div style="font-size: 11px; color: #64748B; font-weight: 500;">
                        <span>{accuracy_sub}</span>
                    </div>
                </div>
            </div>
            """)

        with c_card2:
            render_html(f"""
            <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 14px; padding: 16px 14px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12); position: relative; overflow: hidden; min-height: 148px; display: flex; flex-direction: column; justify-content: space-between; margin-top: 14px; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;">
                <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #EF4444, #DC2626);"></div>
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">ERROR (%)</span>
                        <div style="width: 28px; height: 28px; border-radius: 8px; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.35); display: flex; align-items: center; justify-content: center;">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                                <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
                                <line x1="12" y1="9" x2="12" y2="13"></line>
                                <line x1="12" y1="17" x2="12.01" y2="17"></line>
                            </svg>
                        </div>
                    </div>
                    <div style="display: flex; align-items: baseline; margin-bottom: 10px;">
                        <span style="font-size: 26px; font-weight: 900; color: #EF4444; letter-spacing: -0.5px; line-height: 1;">{error_str}</span>
                    </div>
                </div>
                <div>
                    <div style="width: 100%; height: 7px; background: #1E293B; border-radius: 9999px; overflow: hidden; margin-bottom: 6px;">
                        <div style="width: {error_pct}%; height: 100%; background: linear-gradient(90deg, #EF4444, #F87171); border-radius: 9999px;"></div>
                    </div>
                    <div style="font-size: 11px; color: #64748B; font-weight: 500;">
                        <span>{error_sub}</span>
                    </div>
                </div>
            </div>
            """)

        with c_card3:
            render_html(f"""
            <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 14px; padding: 16px 14px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12); position: relative; overflow: hidden; min-height: 148px; display: flex; flex-direction: column; justify-content: space-between; margin-top: 14px; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;">
                <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #F59E0B, #D97706);"></div>
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;" title="Pending Verification (%)">PENDING VERIF. (%)</span>
                        <div style="width: 28px; height: 28px; border-radius: 8px; background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.35); display: flex; align-items: center; justify-content: center;">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="10"></circle>
                                <polyline points="12 6 12 12 16 14"></polyline>
                            </svg>
                        </div>
                    </div>
                    <div style="display: flex; align-items: baseline; margin-bottom: 10px;">
                        <span style="font-size: 26px; font-weight: 900; color: #F59E0B; letter-spacing: -0.5px; line-height: 1;">{pending_str}</span>
                    </div>
                </div>
                <div>
                    <div style="width: 100%; height: 7px; background: #1E293B; border-radius: 9999px; overflow: hidden; margin-bottom: 6px;">
                        <div style="width: {pending_pct}%; height: 100%; background: linear-gradient(90deg, #F59E0B, #FBBF24); border-radius: 9999px;"></div>
                    </div>
                    <div style="font-size: 11px; color: #64748B; font-weight: 500;">
                        <span>{pending_sub}</span>
                    </div>
                </div>
            </div>
            """)

        # Verification Summary Banner with All 4 Metrics
        total_det = metrics.get('total_detected_count', 0)
        render_html(f"""
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 12px 16px; margin-top: 12px; display: flex; justify-content: space-between; align-items: center; font-family: 'Inter', -apple-system, sans-serif;">
            <div>
                <span style="color: #94A3B8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Field Verification Summary</span>
                <div style="color: #F8FAFC; font-size: 13.5px; font-weight: 700; margin-top: 2px;">
                    <b>{total_det}</b> of {total_fields} fields detected &nbsp;•&nbsp; 
                    <span style="color: #10B981;">{verified_correct_count} Auto-Verified</span> &nbsp;|&nbsp; 
                    <span style="color: #EF4444;">{incorrect_count} Incorrect</span> &nbsp;|&nbsp; 
                    <span style="color: #F59E0B;">{needs_verification_count} Require Manual Verification</span>
                </div>
            </div>
            <div style="text-align: right;">
                <span style="color: #94A3B8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Accuracy Rate</span>
                <div style="color: #10B981; font-size: 14px; font-weight: 800; margin-top: 2px;">
                    {accuracy_str}
                </div>
            </div>
        </div>
        """)

        # -------------------------------------------------------------
        # FIELD-LEVEL STATUS BREAKDOWN (ALL 11 FIELDS)
        # -------------------------------------------------------------
        with st.expander("📊 **View Status for All 11 Fields**", expanded=False):
            field_table_rows = []
            for idx, f in enumerate(COVER_FIELDS, 1):
                fid = f["id"]
                label = f["label"]
                field_state = v_map.get(fid, {})
                is_ver = field_state.get("verified", False)
                ocr_val = result["initial_extracted_fields"].get(fid, "")
                gt_val = field_state.get("ground_truth", "")

                disp_ocr = val_clean(ocr_val, "—")
                disp_gt = val_clean(gt_val, "—") if is_ver else "<span style='color:#94A3B8; font-style:italic;'>Unverified</span>"
                f_stat = field_status_map.get(fid, "Needs Verification")

                if f_stat == "Needs Verification":
                    st_badge = "<span style='color:#B45309; font-weight:600; font-size:11px; background:#FEF3C7; padding:2px 6px; border-radius:10px;'>⚠️ Needs Verification</span>"
                elif f_stat == "Verified Correct":
                    st_badge = "<span style='color:#16A34A; font-weight:600; font-size:11px; background:#DCFCE7; padding:2px 6px; border-radius:10px;'>✓ Verified Correct</span>"
                else:
                    st_badge = "<span style='color:#DC2626; font-weight:600; font-size:11px; background:#FEE2E2; padding:2px 6px; border-radius:10px;'>✗ Incorrect</span>"

                field_table_rows.append(f"""
                <tr style="border-bottom: 1px solid #F1F5F9; font-size: 12px;">
                    <td style="padding: 7px 5px; color: #64748B; font-weight: 600;">{idx}</td>
                    <td style="padding: 7px 5px; color: #0F172A; font-weight: 600;">{label}</td>
                    <td style="padding: 7px 5px; color: #334155; font-family: monospace; font-size: 11.5px;">{disp_ocr}</td>
                    <td style="padding: 7px 5px; color: #0F172A; font-size: 11.5px;">{disp_gt}</td>
                    <td style="padding: 7px 5px; text-align: center;">{st_badge}</td>
                </tr>
                """)

            all_field_rows_str = "".join(field_table_rows)
            render_html(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; overflow-x: auto; margin-top: 10px; padding: 10px 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <table style="width: 100%; border-collapse: collapse; text-align: left; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;">
                    <thead>
                        <tr style="border-bottom: 2px solid #E2E8F0; font-size: 11px; color: #64748B; text-transform: uppercase; letter-spacing: 0.3px;">
                            <th style="padding: 5px;">#</th>
                            <th style="padding: 5px;">Field Name</th>
                            <th style="padding: 5px;">OCR Extracted</th>
                            <th style="padding: 5px;">Ground Truth</th>
                            <th style="padding: 5px; text-align: center;">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {all_field_rows_str}
                    </tbody>
                    <tfoot>
                        <tr style="border-top: 2px solid #CBD5E1; font-size: 12px; font-weight: 700; background: #F8FAFC;">
                            <td colspan="3" style="padding: 8px 5px; color: #0F172A;">Total: {total_fields} Fields</td>
                            <td colspan="2" style="padding: 8px 5px; text-align: right; color: #334155;">
                                <span style="color:#10B981;">{verified_correct_count} Correct ({accuracy_str})</span> &nbsp;|&nbsp; 
                                <span style="color:#EF4444;">{incorrect_count} Incorrect ({error_str})</span> &nbsp;|&nbsp; 
                                <span style="color:#F59E0B;">{needs_verification_count} Pending ({pending_str})</span>
                            </td>
                        </tr>
                    </tfoot>
                </table>
            </div>
            """)

    # -------------------------------------------------------------
    # VERIFICATION & QUICK CORRECTION (WITH INLINE ORIGINAL CROPS)
    # -------------------------------------------------------------
    st.markdown("---")
    fields_needing_review = [
        f for f in COVER_FIELDS
        if field_status_map.get(f["id"]) in ["Needs Verification", "Needs Manual Verification", "Pending", "Incorrect"]
    ]

    with st.expander("✏️ **Reviewer Verification & Correction Controls**", expanded=bool(fields_needing_review)):
        if not fields_needing_review:
            st.success("🎉 **All 11 student fields have been verified against the original PDF!** No manual correction is required.")
        else:
            st.info(f"ℹ️ **{len(fields_needing_review)} field(s) require manual review** (blank, missing, or uncertain in PDF). Auto-verified fields ({verified_correct_count}) do not require manual intervention.")

        # Batch quick action buttons
        c_qa1, c_qa2, c_qa3 = st.columns([1.5, 1.5, 1.5])
        with c_qa1:
            if st.button("✓ Confirm All Detected", key=f"btn_bulk_confirm_{student_uid}", type="primary", width="stretch", help="Confirm all detected fields as verified correct"):
                for f in COVER_FIELDS:
                    fid = f["id"]
                    ocr_val = result["initial_extracted_fields"].get(fid, "")
                    is_missing = not ocr_val or str(ocr_val).strip().lower() in ["not detected", "not_detected", "none", "not present", "needs review", ""]
                    if not is_missing:
                        v_map[fid]["status"] = "Verified"
                        v_map[fid]["verified"] = True
                        v_map[fid]["is_explicit_incorrect"] = False
                        w_key = f"edit_gt_{fid}_{student_uid}"
                        if w_key in st.session_state and st.session_state[w_key].strip():
                            v_map[fid]["ground_truth"] = st.session_state[w_key].strip()
                        else:
                            v_map[fid]["ground_truth"] = ocr_val
                try:
                    up_metrics = calculate_cover_field_metrics(COVER_FIELDS, v_map, result["initial_extracted_fields"])
                    db.save_field_verifications(student_uid, result.get("filename", "booklet"), v_map, up_metrics, raw_roll)
                except Exception:
                    pass
                st.rerun()
        with c_qa2:
            if st.button("⚠️ Reset to Needs Review", key=f"btn_bulk_unver_{student_uid}", type="secondary", width="stretch"):
                for f in COVER_FIELDS:
                    fid = f["id"]
                    v_map[fid]["status"] = "Needs Verification"
                    v_map[fid]["verified"] = False
                    v_map[fid]["is_explicit_incorrect"] = False
                try:
                    up_metrics = calculate_cover_field_metrics(COVER_FIELDS, v_map, result["initial_extracted_fields"])
                    db.save_field_verifications(student_uid, result.get("filename", "booklet"), v_map, up_metrics, raw_roll)
                except Exception:
                    pass
                st.rerun()
        with c_qa3:
            if st.button("🔄 Auto-Verify Remaining", key=f"btn_bulk_reset_{student_uid}", type="secondary", width="stretch"):
                trusted_roster = st.session_state.get("trusted_reference_data", {})
                matching_ref = (
                    trusted_roster.get(clean_roll) or
                    trusted_roster.get(raw_roll) or
                    trusted_roster.get(f"student_{cur_idx + 1}") or
                    {}
                )
                auto_v_map = perform_automatic_verification(
                    initial_extracted_fields=result["initial_extracted_fields"],
                    crops_dict=crops_dict,
                    cover_fields=COVER_FIELDS,
                    trusted_reference=matching_ref if matching_ref else None
                )
                st.session_state[ver_state_key] = auto_v_map
                for f in COVER_FIELDS:
                    st.session_state.pop(f"edit_gt_{f['id']}_{student_uid}", None)
                try:
                    up_metrics = calculate_cover_field_metrics(COVER_FIELDS, auto_v_map, result["initial_extracted_fields"])
                    db.save_field_verifications(student_uid, result.get("filename", "booklet"), auto_v_map, up_metrics, raw_roll)
                except Exception:
                    pass
                st.rerun()

        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        # Show only fields requiring review
        if fields_needing_review:
            col_f1, col_f2 = st.columns(2)
            half = (len(fields_needing_review) + 1) // 2
            for i, f in enumerate(fields_needing_review):
                target_col = col_f1 if i < half else col_f2
                with target_col:
                    fid = f["id"]
                    flabel = f["label"]
                    f_state = v_map.get(fid, {})
                    f_cur_gt = f_state.get("ground_truth", "") or result["initial_extracted_fields"].get(fid, "")
                    f_stat = field_status_map.get(fid, "Needs Verification")
                    f_reason = f_state.get("reason", "Requires manual review")

                    # Crop image preview
                    c_data = crops_dict.get(fid, {})
                    crop_img = c_data.get("crop_image")

                    if f_stat == "Verified Correct" or f_stat == "Verified":
                        st_pill = "<span style='color:#16A34A; font-weight:700; font-size:11px; background:#DCFCE7; padding:2px 7px; border-radius:8px;'>✓ Verified</span>"
                    elif f_stat == "Incorrect":
                        st_pill = "<span style='color:#DC2626; font-weight:700; font-size:11px; background:#FEE2E2; padding:2px 7px; border-radius:8px;'>✗ Incorrect</span>"
                    else:
                        st_pill = "<span style='color:#B45309; font-weight:700; font-size:11px; background:#FEF3C7; padding:2px 7px; border-radius:8px;'>⚠️ Needs Verification</span>"

                    render_html(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                        <span style="font-size:13px; font-weight:700; color:#0F172A;">{flabel}</span>
                        <div>{st_pill}</div>
                    </div>
                    <div style="font-size:11px; color:#64748B; margin-bottom:6px;">{f_reason}</div>
                    """)

                    if crop_img:
                        st.image(crop_img, width=220)

                    w_key = f"edit_gt_{fid}_{student_uid}"
                    val_entered = st.text_input(
                        f"Edit {flabel}",
                        value=f_cur_gt,
                        key=w_key,
                        placeholder=f["placeholder"],
                        label_visibility="collapsed"
                    )

                    b_c1, b_c2, b_c3 = st.columns([1.2, 1.2, 1.2])
                    with b_c1:
                        if st.button("✓ Confirm", key=f"btn_cf_{fid}_{student_uid}", help=f"Confirm {flabel} as correct"):
                            v_map[fid]["ground_truth"] = val_entered.strip()
                            v_map[fid]["status"] = "Verified"
                            v_map[fid]["verified"] = True
                            v_map[fid]["is_explicit_incorrect"] = False
                            try:
                                up_m = calculate_cover_field_metrics(COVER_FIELDS, v_map, result["initial_extracted_fields"])
                                db.save_field_verifications(student_uid, result.get("filename", "booklet"), v_map, up_m, raw_roll)
                            except Exception:
                                pass
                            st.rerun()
                    with b_c2:
                        if st.button("✗ Incorrect", key=f"btn_bad_{fid}_{student_uid}", help=f"Mark {flabel} as incorrect"):
                            v_map[fid]["status"] = "Incorrect"
                            v_map[fid]["verified"] = True
                            v_map[fid]["is_explicit_incorrect"] = True
                            try:
                                up_m = calculate_cover_field_metrics(COVER_FIELDS, v_map, result["initial_extracted_fields"])
                                db.save_field_verifications(student_uid, result.get("filename", "booklet"), v_map, up_m, raw_roll)
                            except Exception:
                                pass
                            st.rerun()
                    with b_c3:
                        if st.button("⚠️ Pending", key=f"btn_pen_{fid}_{student_uid}", help=f"Set {flabel} to Needs Verification"):
                            v_map[fid]["status"] = "Needs Verification"
                            v_map[fid]["verified"] = False
                            v_map[fid]["is_explicit_incorrect"] = False
                            try:
                                up_m = calculate_cover_field_metrics(COVER_FIELDS, v_map, result["initial_extracted_fields"])
                                db.save_field_verifications(student_uid, result.get("filename", "booklet"), v_map, up_m, raw_roll)
                            except Exception:
                                pass
                            st.rerun()

                    st.markdown("<div style='margin-bottom: 12px; border-bottom: 1px dashed #E2E8F0;'></div>", unsafe_allow_html=True)

        st.markdown("---")
        c_act1, c_act2 = st.columns([2, 1])
        with c_act1:
            if st.button("💾 Confirm Details & Save Evaluation", key=f"btn_confirm_save_{student_uid}", type="primary", width="stretch"):
                # Mark all fields as verified for this student
                for f in COVER_FIELDS:
                    fid = f["id"]
                    v_map[fid]["verified"] = True
                    v_map[fid]["is_explicit_incorrect"] = False
                    w_key = f"edit_gt_{fid}_{student_uid}"
                    if w_key in st.session_state and st.session_state[w_key].strip():
                        v_map[fid]["ground_truth"] = st.session_state[w_key].strip()

                student_info.student_name = v_map["student_name"].get("ground_truth", "")
                student_info.roll_number = v_map["enrollment_no"].get("ground_truth", "")
                student_info.registration_number = v_map["enrollment_no"].get("ground_truth", "")
                student_info.test_number = v_map["test_no"].get("ground_truth", "")
                student_info.section = v_map["section"].get("ground_truth", "")
                student_info.exam_date = v_map["date"].get("ground_truth", "")
                student_info.class_year = v_map["class"].get("ground_truth", "")
                student_info.course_title = v_map["course_title"].get("ground_truth", "")
                student_info.subject_name = v_map["course_title"].get("ground_truth", "")
                student_info.subject_code = v_map["course_code"].get("ground_truth", "")
                student_info.semester = v_map["semester"].get("ground_truth", "")
                if v_map["instructor_name"].get("ground_truth"):
                    setattr(student_info, "instructor_name", v_map["instructor_name"].get("ground_truth"))
                if v_map["grand_total"].get("ground_truth"):
                    setattr(student_info, "grand_total", v_map["grand_total"].get("ground_truth"))

                eval_rec.student = student_info
                eval_rec.is_approved = True

                if result.get("saved_eval_id"):
                    eval_rec.id = result["saved_eval_id"]
                    db.update_evaluation(eval_rec)
                    saved_id = eval_rec.id
                else:
                    saved_id = db.save_evaluation(eval_rec)
                    eval_rec.id = saved_id

                result["is_saved"] = True
                result["saved_eval_id"] = saved_id
                result["student_details"] = student_info
                result["evaluation_record"] = eval_rec
                if student_records is not None and cur_idx < len(student_records):
                    student_records[cur_idx] = result
                    st.session_state["student_records"] = student_records
                st.session_state["evaluation_result"] = result
                st.success(f"✅ Details verified and evaluation record #{saved_id} successfully saved to SQLite!")
                st.rerun()

        with c_act2:
            if st.button("🔄 Re-run OCR on This Page", key=f"btn_rerun_p1_{student_uid}", width="stretch"):
                if first_page_img:
                    with st.spinner("Re-processing Page template OCR..."):
                        info, raw_t, conf_score = process_first_page(first_page_img)
                        result["student_details"] = info
                        eval_rec.student = info
                        if "initial_extracted_fields" in result:
                            del result["initial_extracted_fields"]
                        if ver_state_key in st.session_state:
                            del st.session_state[ver_state_key]
                        if student_records is not None and cur_idx < len(student_records):
                            student_records[cur_idx] = result
                            st.session_state["student_records"] = student_records
                        st.session_state["evaluation_result"] = result
                        st.rerun()

    st.markdown("---")

    # -------------------------------------------------------------
    # SECTION 2: SIGNATURE DETECTION STATUS (FIELDS 16, 17, 18)
    # -------------------------------------------------------------
    sig_status = telemetry.get("signatures_status", {
        "student_signature": "Detected / Available" if getattr(student_info, "student_signature_present", True) else "Not Detected",
        "invigilator_signature": "Detected / Available" if getattr(student_info, "invigilator_signature_present", True) else "Not Detected",
        "examiner_signature": "Detected / Available" if getattr(student_info, "examiner_signature_present", True) else "Not Detected"
    })

    st.write("##### ✍️ SIGNATURE DETECTION STATUS")
    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        st.info(f"**16. Student Signature:** {sig_status.get('student_signature', 'Detected / Available')}")
    with c_s2:
        st.info(f"**17. Invigilator Signature:** {sig_status.get('invigilator_signature', 'Detected / Available')}")
    with c_s3:
        st.info(f"**18. Examiner Signature:** {sig_status.get('examiner_signature', 'Detected / Available')}")

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # SECTION 3: QUESTION-WISE MARKS BREAKDOWN TABLE
    # -------------------------------------------------------------
    st.write("##### QUESTION-WISE MARKS BREAKDOWN TABLE (Q1 to Q10)")
    if marks_grid_results:
        def fmt_val(v):
            if v is None:
                return "—"
            if isinstance(v, (int, float)):
                return f"{v:g}"
            return str(v)

        grid_rows = []
        for item in marks_grid_results:
            sub = item.get("sub_marks", {})
            calc_t = item.get("calculated_marks", 0.0)
            ex_t = item.get("examiner_total")
            
            val_status = "Matched ✓"
            if ex_t is not None and abs(ex_t - calc_t) > 0.1:
                val_status = "Review Difference ⚠️"

            grid_rows.append({
                "Q.No": str(item.get("question_number")),
                "A": fmt_val(sub.get("A")),
                "B": fmt_val(sub.get("B")),
                "C": fmt_val(sub.get("C")),
                "D": fmt_val(sub.get("D")),
                "E": fmt_val(sub.get("E")),
                "Extracted Total": fmt_val(ex_t),
                "Calculated Total": fmt_val(calc_t),
                "Validation": val_status
            })
        st.dataframe(pd.DataFrame(grid_rows), width="stretch")

    # -------------------------------------------------------------
    # SECTION 4: GRAND TOTAL VALIDATION
    # -------------------------------------------------------------
    calc_grand = eval_rec.final_total_marks
    written_totals = [item.get("examiner_total") for item in marks_grid_results if item.get("examiner_total") is not None]
    written_grand = round(sum(written_totals), 1) if written_totals else calc_grand
    diff = round(abs(calc_grand - written_grand), 1)

    if calc_grand > eval_rec.total_max_marks:
        grand_status = "Exceeds Max Exam Marks ⚠️"
    elif diff == 0:
        grand_status = "Matched ✓"
    else:
        grand_status = f"Difference Detected ({diff} marks) ⚠️"

    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; color: #0F172A; padding: 18px 22px; border-radius: 12px; margin: 20px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:16px;">
            <span><b>Extracted Examiner Grand Total:</b> {written_grand:g}</span>
            <span><b>Calculated Grand Total:</b> {calc_grand:g} / {eval_rec.total_max_marks:g}</span>
            <span style="font-weight:bold; color:{'#16A34A' if grand_status=='Matched ✓' else '#D97706'};"><b>Validation Status:</b> {grand_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # PROCESSING TIME INFORMATION
    # -------------------------------------------------------------
    if step_timings:
        st.write("##### ⏱️ Pipeline Timing Breakdown")
        st.markdown(f"""
        • **Orientation & Normalization:** `{step_timings.get('orientation_time', 0.0)}s`  
        • **Student Details OCR:** `{step_timings.get('student_ocr_time', 0.0)}s`  
        • **Marks Table Grid OCR:** `{step_timings.get('marks_table_time', 0.0)}s`  
        • **Answer Extraction:** `{step_timings.get('answer_ocr_time', 0.0)}s`  
        • **Evaluation & Grading:** `{step_timings.get('evaluation_time', 0.0)}s`  
        • **Total Student Processing Time:** `{step_timings.get('total_time', 0.0)}s`
        """)

    st.markdown("---")

    # -------------------------------------------------------------
    # SIDE-BY-SIDE ANSWER REVIEW & TEACHER OVERRIDE
    # -------------------------------------------------------------
    st.write("##### Side-by-Side Review & Teacher Mark Overrides")
    st.info("ℹ️ AI marks are suggestions. You can adjust the final marks or comments for any question below to recalculate score, grade, and pass/fail.")
    
    updated_questions = []
    new_total_final = 0.0

    for idx, q in enumerate(eval_rec.questions):
        with st.expander(f"📑 Question {q.question_id} | AI Suggested: {q.ai_marks}/{q.max_marks} | Teacher Final: {q.final_marks}/{q.max_marks}", expanded=True):
            c_img, c_txt = st.columns([1, 1])

            with c_img:
                st.write("**Original Answer Sheet Snippet**")
                if answer_page_imgs:
                    st.image(answer_page_imgs[min(idx, len(answer_page_imgs)-1)], caption=f"{q.question_id} Answer Page", width="stretch")
                else:
                    st.caption("No page image snippet available.")

            with c_txt:
                st.write(f"**Extracted Student Answer ({q.question_id})**")
                st.info(q.student_answer or "[Not Answered]")

                st.write("**Reference Model Answer**")
                st.caption(q.model_answer or "Model answer unavailable")

                c_m1, c_comm = st.columns([1, 2])
                with c_m1:
                    final_input = st.number_input(
                        f"Final Marks ({q.question_id})",
                        min_value=0.0,
                        max_value=float(q.max_marks),
                        value=float(q.final_marks),
                        step=0.5,
                        key=f"auto_m_{q.question_id}_{idx}_{student_uid}"
                    )
                with c_comm:
                    comm_input = st.text_input(
                        "Teacher Comments",
                        value=q.teacher_comments or "",
                        key=f"auto_c_{q.question_id}_{idx}_{student_uid}"
                    )

                q.final_marks = final_input
                q.teacher_comments = comm_input
                new_total_final += final_input
                updated_questions.append(q)

    if st.button("✅ APPROVE & SAVE FINAL RESULT", key=f"btn_approve_save_{student_uid}", type="primary", width="stretch"):
        eval_rec.questions = updated_questions
        eval_rec.final_total_marks = round(new_total_final, 1)
        eval_rec.percentage = round((new_total_final / eval_rec.total_max_marks * 100.0), 2) if eval_rec.total_max_marks > 0 else 0.0
        eval_rec.grade, _ = determine_grade_and_status(eval_rec.percentage, sys_settings)
        eval_rec.is_approved = True

        if eval_rec.id:
            db.update_evaluation(eval_rec)
            saved_id = eval_rec.id
        else:
            saved_id = db.save_evaluation(eval_rec)
            eval_rec.id = saved_id
            result["saved_eval_id"] = saved_id

        result["is_saved"] = True
        result["evaluation_record"] = eval_rec
        if student_records is not None and cur_idx < len(student_records):
            student_records[cur_idx] = result
            st.session_state["student_records"] = student_records
        st.session_state["evaluation_result"] = result
        st.success(f"Final score approved and recorded in SQLite database! (Record #{saved_id})")
        st.rerun()

    # -------------------------------------------------------------
    # SECTION 6: DIAGNOSTICS & RAW OCR LOGS
    # -------------------------------------------------------------
    st.markdown("---")
    with st.expander("🛠️ **Raw OCR Diagnostics & Template Overlays**", expanded=False):
        if telemetry and "overlay_image" in telemetry:
            st.write("**Master Booklet Template Overlay (1600 x 2200 Canvas):**")
            st.image(telemetry["overlay_image"], caption="Green bounding boxes indicate detected value regions", width="stretch")

        if table_overlay_img:
            st.write("**Examiner Marks Table Grid Overlay:**")
            st.image(table_overlay_img, caption="Green grid cells indicate examiner mark extraction zones", width="stretch")

        st.write("**Telemetry JSON:**")
        st.json({k: v for k, v in telemetry.items() if k not in ["crops", "overlay_image"]})

        st.write("**Raw OCR Output Text:**")
        st.code(result.get("raw_ocr_debug_text", "No raw OCR text available."), language="text")


def render_upload_page(db: EvaluationDatabase):
    st.subheader("Automated Student Answer Booklet Evaluator")
    st.write(
        "Upload scanned student answer booklets (Multi-Page PDF, Single-Student PDF, or JPG/PNG images). "
        "The system will automatically detect page orientation, extract student information for each page, "
        "evaluate answers, calculate grades, and allow seamless Next/Previous navigation across students."
    )

    sys_settings = db.get_settings()
    current_ocr = getattr(sys_settings, "ocr_engine", "paddleocr") or "paddleocr"

    ocr_names = {
        "paddleocr": "PaddleOCR + OpenCV (High Accuracy Deep Learning - Recommended)",
        "yolo_paddleocr": "YOLOv8 + PaddleOCR (Region Guided Deep Learning)",
        "yolo_easyocr": "YOLOv8 + EasyOCR (Handwritten Sheets)",
        "yolo_ocr": "YOLOv8 + Tesseract (ROI Guided OCR)",
        "hybrid": "Hybrid Ensemble (PaddleOCR + EasyOCR + Tesseract Voting)",
        "easyocr": "EasyOCR Deep Learning (Handwriting Optimized)",
        "tesseract": "Tesseract OCR (Standard Printed Baseline)"
    }
    if current_ocr not in ocr_names:
        current_ocr = "paddleocr"

    c_ocr1, c_ocr2 = st.columns([2, 1])
    with c_ocr1:
        st.info(f"⚡ **Active OCR Engine:** `{ocr_names.get(current_ocr, current_ocr.upper())}`")
    with c_ocr2:
        new_eng_disp = st.selectbox(
            "Switch Engine",
            options=list(ocr_names.values()),
            index=list(ocr_names.keys()).index(current_ocr) if current_ocr in ocr_names else 0,
            label_visibility="collapsed"
        )
        selected_key = [k for k, v in ocr_names.items() if v == new_eng_disp][0]
        if selected_key != current_ocr:
            sys_settings.ocr_engine = selected_key
            db.update_settings(sys_settings)
            st.rerun()

    # Session state initialization
    if "student_records" not in st.session_state:
        st.session_state["student_records"] = []
    if "current_student_index" not in st.session_state:
        st.session_state["current_student_index"] = 0
    if "processed_upload_key" not in st.session_state:
        st.session_state["processed_upload_key"] = None
    if "evaluation_result" not in st.session_state:
        st.session_state["evaluation_result"] = None

    # Processing mode selector
    c_mode1, c_mode2 = st.columns([2.5, 1.5])
    with c_mode1:
        doc_mode = st.radio(
            "📄 Document Structure Mode:",
            options=[
                "👥 Multi-Student Mode (Each page is a different student's answer sheet - Recommended)",
                "📖 Single-Student Booklet Mode (Page 1 = details, Pages 2..N = answers for 1 student)"
            ],
            index=0,
            horizontal=True
        )
    is_multi_student = "Multi-Student" in doc_mode

    with c_mode2:
        records_list = st.session_state.get("student_records", [])
        if records_list:
            tot = len(records_list)
            c_idx = st.session_state.get("current_student_index", 0)
            st.metric("Total Students Detected", f"{tot} Student{'s' if tot != 1 else ''}", f"Viewing #{c_idx + 1}")

    uploaded_files = st.file_uploader(
        "Upload Scanned Answer Sheets (Multi-Page PDF or Images)",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload a single PDF containing 5-6 student pages, or multiple PDF/image files."
    )

    with st.expander("📋 **Supply Trusted Reference Roster / Ground Truth (Optional)**", expanded=False):
        st.caption("Upload an official student roster or benchmark ground-truth file (CSV or JSON) to automatically verify OCR results against trusted reference data.")
        ref_file = st.file_uploader(
            "Upload Reference Roster (CSV / JSON)",
            type=["csv", "json"],
            key="uploader_trusted_reference_file",
            help="Supported columns / keys: enrollment_no, student_name, test_no, section, date, class, course_title, course_code, semester, instructor_name, grand_total"
        )
        if ref_file is not None:
            try:
                import json as _json, csv as _csv, io as _io
                content_bytes = ref_file.getvalue()
                roster_map = {}
                if ref_file.name.endswith(".json"):
                    data = _json.loads(content_bytes.decode("utf-8", errors="ignore"))
                    samples = data.get("samples", data if isinstance(data, list) else [data])
                    for s in samples:
                        gt = s.get("ground_truth", s)
                        roll = re.sub(r"[^A-Za-z0-9]", "", str(gt.get("enrollment_no") or gt.get("roll_number") or "")).upper()
                        if roll:
                            roster_map[roll] = gt
                else:
                    reader = _csv.DictReader(_io.StringIO(content_bytes.decode("utf-8", errors="ignore")))
                    for row in reader:
                        roll = re.sub(r"[^A-Za-z0-9]", "", str(row.get("enrollment_no") or row.get("roll_number") or "")).upper()
                        if roll:
                            roster_map[roll] = row
                st.session_state["trusted_reference_data"] = roster_map
                st.success(f"✓ Loaded {len(roster_map)} trusted student reference record(s). Matching answer sheets will be auto-verified against this reference.")
            except Exception as _e:
                st.error(f"Error parsing reference file: {_e}")

    if uploaded_files:
        # Build batch fingerprint from all uploaded files
        batch_key = "_".join([f"{getattr(f, 'name', 'file')}_{getattr(f, 'size', 0)}" for f in uploaded_files]) + f"_{is_multi_student}"
        is_new_upload = st.session_state.get("processed_upload_key") != batch_key

        if is_new_upload:
            progress_box = st.empty()
            progress_bar = st.progress(0)
            try:
                progress_box.info(f"📁 Reading {len(uploaded_files)} file(s)...")
                records = process_multi_student_booklet_files(
                    uploaded_files=uploaded_files,
                    db=db,
                    progress_box=progress_box,
                    progress_bar=progress_bar,
                    is_multi_student_mode=is_multi_student
                )
                st.session_state["student_records"] = records
                st.session_state["current_student_index"] = 0
                st.session_state["processed_upload_key"] = batch_key
                if records:
                    st.session_state["evaluation_result"] = records[0]

                progress_box.empty()
                progress_bar.empty()
                st.success(f"✅ Successfully processed {len(records)} student answer sheets!")
                st.rerun()
            except Exception as e:
                progress_box.empty()
                progress_bar.empty()
                st.error(f"❌ **Upload Processing Failed:** {e}")

    student_records = st.session_state.get("student_records", [])
    if student_records:
        total_students = len(student_records)
        cur_idx = st.session_state.get("current_student_index", 0)
        cur_idx = max(0, min(cur_idx, total_students - 1))
        st.session_state["current_student_index"] = cur_idx
        st.session_state["evaluation_result"] = student_records[cur_idx]

        # Render Next / Previous Navigation Bar
        render_student_navigation(student_records)

        # Render selected student's result
        render_student_result(
            result=student_records[cur_idx],
            db=db,
            student_records=student_records,
            cur_idx=cur_idx
        )
    elif st.session_state.get("evaluation_result"):
        # Single record fallback
        render_student_result(
            result=st.session_state["evaluation_result"],
            db=db,
            student_records=[st.session_state["evaluation_result"]],
            cur_idx=0
        )
