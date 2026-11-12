import os
import sys
sys.path.insert(0, ".")
import cv2
import pytesseract
import numpy as np
from PIL import Image
from template_config import BOOKLET_TEMPLATE, NORMALIZED_WIDTH, NORMALIZED_HEIGHT
from ocr.preprocessing import convert_to_opencv, find_best_orthogonal_orientation
from ocr.marks import parse_mark_value

def extract_red_ink_mask(crop_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 65, 45])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([160, 65, 45])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return mask1 | mask2

img_path = r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg"
pil_img = Image.open(img_path)
img_bgr = convert_to_opencv(pil_img)
rot_bgr, rot_angle = find_best_orthogonal_orientation(img_bgr)
norm_bgr = cv2.resize(rot_bgr, (NORMALIZED_WIDTH, NORMALIZED_HEIGHT), interpolation=cv2.INTER_CUBIC)

tbl_x, tbl_y, tbl_w, tbl_h = BOOKLET_TEMPLATE.get("marks_table", (80, 1070, 1440, 980))
table_crop = norm_bgr[tbl_y:tbl_y+tbl_h, tbl_x:tbl_x+tbl_w]
th, tw = table_crop.shape[:2]

col_bounds = {
    "A": (int(tw * 0.06), int(tw * 0.11)),
    "B": (int(tw * 0.11), int(tw * 0.16)),
    "C": (int(tw * 0.16), int(tw * 0.21)),
    "D": (int(tw * 0.21), int(tw * 0.26)),
    "E": (int(tw * 0.26), int(tw * 0.31)),
    "Total": (int(tw * 0.31), int(tw * 0.38))
}

row_height = (th * 0.82) / 10.0
header_offset = th * 0.14

for q_idx in range(1, 11):
    ry_start = int(header_offset + (q_idx - 1) * row_height)
    ry_end = int(header_offset + q_idx * row_height)
    for col_name, (cx_start, cx_end) in col_bounds.items():
        cell_crop = table_crop[ry_start:ry_end, cx_start:cx_end]
        red_mask = extract_red_ink_mask(cell_crop)
        cnt = cv2.countNonZero(red_mask)
        if cnt >= 15:
            inv_red = 255 - red_mask
            inv_red_padded = cv2.copyMakeBorder(inv_red, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
            inv_red_up = cv2.resize(inv_red_padded, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            pil_mask = Image.fromarray(inv_red_up)
            txt = pytesseract.image_to_string(pil_mask, config="--psm 7 -c tessedit_char_whitelist=0123456789/½¼¾.").strip()
            val = parse_mark_value(txt, is_subpart=(col_name!="Total"))
            print(f"Q{q_idx} {col_name}: red_pixels={cnt}, raw_ocr='{txt}', parsed={val}")
        else:
            print(f"Q{q_idx} {col_name}: BLANK (0)")
