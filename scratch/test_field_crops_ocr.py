import sys
import os
sys.path.insert(0, os.path.abspath("."))
import cv2
import re
import numpy as np
from PIL import Image

from template_config import DEFAULT_VALUE_ROIS, NORMALIZED_WIDTH, NORMALIZED_HEIGHT
from ocr.paddle_ocr import get_paddleocr_engine
from ocr.preprocessing import (
    correct_document_perspective,
    remove_shadows_and_enhance,
    find_best_orthogonal_orientation
)

engine = get_paddleocr_engine()

# Load IMG_4867
p = r"data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg"
img = cv2.imread(p)
oriented, ang = find_best_orthogonal_orientation(img, is_first_page=True)
warped, _ = correct_document_perspective(oriented, NORMALIZED_WIDTH, NORMALIZED_HEIGHT)
enhanced = remove_shadows_and_enhance(warped)

print("=== CROPPING EACH FIELD FROM WARPED 1600x2200 CANVAS ===")
for field, (x, y, w, h) in DEFAULT_VALUE_ROIS.items():
    if field in ["marks_table", "examiner_remarks", "recheck_request"]:
        continue
    crop = enhanced[y:y+h, x:x+w]
    txt, conf = engine.read_field_crop(crop)
    print(f"  {field:22s} ({x:4d}, {y:4d}, {w:3d}, {h:3d}): '{txt}' (conf={conf:.1f}%)", flush=True)
