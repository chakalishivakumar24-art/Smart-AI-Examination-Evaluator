import os
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image
from ocr.preprocessing import convert_to_opencv, find_best_orthogonal_orientation
from ocr.first_page_ocr import process_first_page
from ocr.marks_table import extract_marks_table
from template_config import NORMALIZED_WIDTH, NORMALIZED_HEIGHT

img_path = r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg"
print(f"Loading: {img_path}")
pil_img = Image.open(img_path)
img_bgr = convert_to_opencv(pil_img)

rot_bgr, rot_angle = find_best_orthogonal_orientation(img_bgr)
print(f"Detected orientation angle: {rot_angle}")

norm_p1_bgr = cv2.resize(rot_bgr, (NORMALIZED_WIDTH, NORMALIZED_HEIGHT), interpolation=cv2.INTER_CUBIC)
norm_p1_pil = Image.fromarray(cv2.cvtColor(norm_p1_bgr, cv2.COLOR_BGR2RGB))

student_details, raw_text, conf = process_first_page(norm_p1_pil, detected_angle=rot_angle)
print("\n--- EXTRACTED STUDENT DETAILS ---")
for k, v in student_details.__dict__.items():
    if k != "telemetry":
        print(f"  {k}: {v}")

print(f"\nOverall confidence: {conf}")

marks_grid_results, grid_total_marks, grid_max_marks, _ = extract_marks_table(norm_p1_bgr)
print("\n--- MARKS TABLE RESULTS ---")
print(f"Grand Total: {grid_total_marks} / {grid_max_marks}")
for item in marks_grid_results:
    print(f"  {item}")

