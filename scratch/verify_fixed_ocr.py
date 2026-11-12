import os
import cv2
import numpy as np
from PIL import Image
from ocr.student_details import extract_student_details_page1

img_path = r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\test_normalized_page.jpg"
if not os.path.exists(img_path):
    print("Test image not found!")
    exit(1)

pil_img = Image.open(img_path)
print("=== EXECUTING FIXED STEP 3 OCR PIPELINE ===")

details, telemetry = extract_student_details_page1(pil_img, detected_angle=90)

print(f"\nAverage Step 3 Confidence: {telemetry.get('average_confidence', 0.0):.1f}%")
print(f"Total Execution Time:       {telemetry.get('execution_time_seconds', 0.0):.2f}s\n")

print("--- EXTRACTED FIELDS TELEMETRY ---")
crops = telemetry.get("crops", {})
for f_name, f_data in crops.items():
    val = f_data.get("value")
    conf = f_data.get("confidence", 0.0)
    status = f_data.get("status", "OK")
    var = f_data.get("variant", "none")
    print(f"  {f_name:24s} | Conf: {conf:5.1f}% | Status: {status:12s} | Variant: {var:10s} | Val: '{val}'")
