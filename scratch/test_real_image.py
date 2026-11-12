import os
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image
from ocr.marks import extract_marks_table, ocr_cell_crop, parse_mark_value
from ocr.preprocessing import convert_to_opencv, find_best_orthogonal_orientation
from template_config import NORMALIZED_WIDTH, NORMALIZED_HEIGHT

img_path = r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg"
if not os.path.exists(img_path):
    print("File not found:", img_path)
    exit(1)

pil_img = Image.open(img_path)
img_bgr = convert_to_opencv(pil_img)
rot_bgr, rot_angle = find_best_orthogonal_orientation(img_bgr)
norm_bgr = cv2.resize(rot_bgr, (NORMALIZED_WIDTH, NORMALIZED_HEIGHT), interpolation=cv2.INTER_CUBIC)

results, total_marks, max_marks, overlay = extract_marks_table(norm_bgr)

print("=== MARKS TABLE RESULTS ===")
for r in results:
    print(f"{r['question_number']}: sub_marks={r['sub_marks']}, calc={r['calculated_marks']}, examiner_total={r['examiner_total']}")

print("---------------------------")
print(f"Total Marks Extracted: {total_marks} / {max_marks}")
