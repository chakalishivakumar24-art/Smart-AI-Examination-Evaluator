import streamlit as st
import json
import os
import cv2
import numpy as np
from PIL import Image
from database.database import EvaluationDatabase
from models.database_models import SystemSettings
from utils.sample_generator import seed_demo_evaluations
from template_config import EXAM_TEMPLATE_PATH, NORMALIZED_WIDTH, NORMALIZED_HEIGHT
from ocr.yolo_ocr import run_yolo_ocr_pipeline, draw_yolo_detections, get_yolo_detector

def render_settings_page(db: EvaluationDatabase):
    st.subheader("Admin Configuration & System Settings")
    sys_settings = db.get_settings()

    tab_config, tab_yolo, tab_template = st.tabs([
        "⚙️ System & Grade Settings",
        "🎯 YOLO vs OCR Benchmark & Detector Inspector",
        "📐 Fixed Template Coordinates Calibration"
    ])

    with tab_config:
        with st.form("settings_form_page"):
            st.write("##### 🤖 Core Evaluation & OCR Engine Settings")
            f_pass = st.number_input("Pass Percentage Threshold (%)", min_value=10.0, max_value=100.0, value=float(sys_settings.pass_percentage), step=1.0)
            f_mode = st.selectbox("AI Evaluation Engine Mode", ["Local AI/NLP", "Advanced LLM"], index=0 if sys_settings.ai_evaluation_mode == "Local AI/NLP" else 1)
            
            ocr_engine_options = {
                "paddleocr": "PaddleOCR + OpenCV (High Accuracy Deep Learning - Recommended)",
                "yolo_paddleocr": "YOLOv8 + PaddleOCR (Region Guided Deep Learning)",
                "yolo_easyocr": "YOLOv8 + EasyOCR (Handwritten Answer Sheets)",
                "yolo_ocr": "YOLOv8 + Tesseract (ROI Region Guided OCR)",
                "hybrid": "Hybrid Ensemble (PaddleOCR + EasyOCR + Tesseract Voting)",
                "easyocr": "EasyOCR Deep Learning (Handwritten Sequence Recognition)",
                "tesseract": "Tesseract OCR (Standard Printed Baseline)"
            }
            
            current_ocr_key = getattr(sys_settings, "ocr_engine", "paddleocr") or "paddleocr"
            if current_ocr_key not in ocr_engine_options:
                current_ocr_key = "paddleocr"

            selected_ocr_display = st.selectbox(
                "OCR Engine Selection for Answer Extraction",
                options=list(ocr_engine_options.values()),
                index=list(ocr_engine_options.keys()).index(current_ocr_key)
            )
            
            # Map display string back to key
            f_ocr = [k for k, v in ocr_engine_options.items() if v == selected_ocr_display][0]

            st.markdown("---")
            st.write("##### 📊 Grade Threshold Configuration (%)")
            cg1, cg2, cg3 = st.columns(3)
            with cg1:
                f_ap = st.number_input("Grade A+ (≥ %)", value=float(sys_settings.grade_a_plus))
                f_a = st.number_input("Grade A (≥ %)", value=float(sys_settings.grade_a))
            with cg2:
                f_b = st.number_input("Grade B (≥ %)", value=float(sys_settings.grade_b))
                f_c = st.number_input("Grade C (≥ %)", value=float(sys_settings.grade_c))
            with cg3:
                f_d = st.number_input("Grade D (≥ %)", value=float(sys_settings.grade_d))
                f_e = st.number_input("Grade E (≥ %)", value=float(sys_settings.grade_e))

            submit_settings = st.form_submit_button("Save Settings", type="primary")

        if submit_settings:
            new_settings = SystemSettings(
                pass_percentage=f_pass, ai_evaluation_mode=f_mode, ocr_engine=f_ocr,
                grade_a_plus=f_ap, grade_a=f_a, grade_b=f_b, grade_c=f_c, grade_d=f_d, grade_e=f_e
            )
            db.update_settings(new_settings)
            st.success(f"System settings updated! Active OCR Engine: **{ocr_engine_options[f_ocr]}**")
            st.rerun()

        st.markdown("---")
        st.write("##### ⚡ Quick Data Seeding")
        if st.button("🚀 Load Sample Model Answers & Evaluation Data", type="secondary"):
            seed_demo_evaluations(db)
            st.success("Loaded sample evaluations!")
            st.rerun()

    with tab_yolo:
        st.write("##### 🔬 YOLOv8 vs OCR Engine Live Benchmark & Region Inspector")
        st.info("Upload any answer sheet page or test image to inspect YOLOv8 region detection bounding boxes and compare extraction accuracy across engines.")

        uploaded_test_file = st.file_uploader("Upload Test Booklet Image for Benchmark", type=["jpg", "jpeg", "png"])
        
        col_y1, col_y2 = st.columns([1, 1])
        with col_y1:
            conf_thresh = st.slider("YOLO Detection Confidence Threshold", min_value=0.05, max_value=0.90, value=0.25, step=0.05)
        with col_y2:
            test_engine = st.selectbox("Benchmark Engine", ["All Engines Side-by-Side", "YOLOv8 + EasyOCR", "YOLOv8 + Tesseract", "Hybrid Ensemble"])

        if uploaded_test_file is not None:
            try:
                pil_img = Image.open(uploaded_test_file).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                with st.spinner("Running YOLOv8 region detection & OCR analysis..."):
                    detector = get_yolo_detector()
                    detections = detector.detect_regions(img_bgr, conf_thresh=conf_thresh)
                    annotated_bgr = draw_yolo_detections(img_bgr, detections)
                    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))

                b_col1, b_col2 = st.columns([1, 1])
                with b_col1:
                    st.image(annotated_pil, caption=f"YOLOv8 Detected Regions ({len(detections)} blocks detected)", use_column_width=True)
                
                with b_col2:
                    st.markdown("##### 📝 OCR Extraction Benchmark Results")
                    if test_engine == "All Engines Side-by-Side":
                        for eng_k in ["tesseract", "easyocr", "yolo_ocr", "yolo_easyocr", "hybrid"]:
                            txt, conf, _ = run_yolo_ocr_pipeline(img_bgr, engine_name=eng_k, conf_thresh=conf_thresh)
                            st.write(f"**Engine: `{eng_k.upper()}`** — Avg Confidence: `{conf}%`")
                            st.text_area(f"Output ({eng_k})", value=txt[:500] if txt else "[No text detected]", height=90, key=f"bench_{eng_k}")
                    else:
                        eng_k = test_engine.lower().replace("+", "_").replace(" ", "")
                        txt, conf, _ = run_yolo_ocr_pipeline(img_bgr, engine_name=eng_k, conf_thresh=conf_thresh)
                        st.metric("OCR Confidence Score", f"{conf}%")
                        st.text_area("Extracted Text", value=txt, height=280)
            except Exception as e:
                st.error(f"Error running benchmark: {e}")

    with tab_template:
        st.write("##### 📐 Fixed Booklet Master Template Calibration (22 ROIs)")
        st.info(f"Canvas Resolution: **{NORMALIZED_WIDTH} x {NORMALIZED_HEIGHT} px**. Coordinates map the exact location of all 22 student/exam details fields and signature zones on Page 1.")
        
        if os.path.exists(EXAM_TEMPLATE_PATH):
            try:
                with open(EXAM_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                    template_data = json.load(f)
                
                st.write("**`templates/exam_template.json` Configuration:**")
                st.json(template_data)
            except Exception as e:
                st.error(f"Error reading JSON template: {e}")
        else:
            st.warning("Template JSON file not found at path.")

    st.caption("SMART AI EXAMINATION EVALUATOR v1.0.0 — Complete Operational Build")
