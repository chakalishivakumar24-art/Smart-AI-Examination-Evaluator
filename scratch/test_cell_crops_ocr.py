import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from ocr.paddle_ocr import get_paddleocr_engine
from PIL import Image

engine = get_paddleocr_engine()
img = cv2.imread(r"scratch\warped_09b450b9_IMG_5774 01.PNG.jpg")

col_names = ["A", "B", "C", "D", "E", "Total"]
col_ranges = [
    (145, 237),
    (237, 329),
    (329, 420),
    (420, 512),
    (512, 600),
    (600, 716)
]

row_ranges = [
    (1168, 1247), # Q1
    (1247, 1328), # Q2
    (1328, 1413), # Q3
    (1413, 1496), # Q4
    (1496, 1578), # Q5
    (1578, 1661), # Q6
    (1661, 1741), # Q7
    (1741, 1818), # Q8
    (1818, 1895), # Q9
    (1895, 1970), # Q10
    (1970, 2038)  # Grand Total row
]

print("=== MARKS CELL CROPS AND OCR ===")
for r_idx, (y1, y2) in enumerate(row_ranges):
    q_label = f"Q{r_idx+1}" if r_idx < 10 else "Grand Total Row"
    row_str = []
    for c_idx, (x1, x2) in enumerate(col_ranges):
        # 5px inner padding to avoid black grid border lines
        cell = img[y1+5:y2-5, x1+5:x2-5]
        
        # Check if cell is blank
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        dark_pixels = np.sum(gray < 180)
        
        # Check red ink
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(mask1 | mask2)
        
        cname = col_names[c_idx]
        if dark_pixels < 25 and red_pixels < 10:
            row_str.append(f"{cname}: [blank]")
        else:
            txt, conf = engine.read_field_crop(cell)
            row_str.append(f"{cname}: '{txt}'(red={red_pixels}, dark={dark_pixels})")
            
    print(f"{q_label:16s} | " + " | ".join(row_str))

# Also inspect Grand total circled region
# On row 10 / Grand total row, Total column: x=600..716, y=1895..2038
gt_crop = img[1895:2038, 550:730]
gt_txt, gt_conf = engine.read_field_crop(gt_crop)
print(f"\nCircled Examiner Grand Total Region ({gt_crop.shape}): '{gt_txt}' (conf={gt_conf}%)")
