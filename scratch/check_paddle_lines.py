import os
import sys
sys.path.insert(0, ".")
import cv2
from PIL import Image, ImageOps
from ocr.paddle_ocr import PaddleOCREngine

img_path = r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg"
engine = PaddleOCREngine()

# Run raw cv2.imread vs ImageOps.exif_transpose
pil_img = Image.open(img_path)
pil_trans = ImageOps.exif_transpose(pil_img)

print("Original PIL size:", pil_img.size)
print("Transposed PIL size:", pil_trans.size)

# Let's run PaddleOCR on the normalized image that extract_student_details_page1 was using!
from ocr.preprocessing import convert_to_opencv, find_best_orthogonal_orientation
from template_config import NORMALIZED_WIDTH, NORMALIZED_HEIGHT

img_bgr = convert_to_opencv(pil_img)
rot_bgr, rot_angle = find_best_orthogonal_orientation(img_bgr)
norm_bgr = cv2.resize(rot_bgr, (NORMALIZED_WIDTH, NORMALIZED_HEIGHT), interpolation=cv2.INTER_CUBIC)

print(f"rot_angle = {rot_angle}")
res = engine.ocr_reader.ocr(norm_bgr, cls=False)
lines = res[0] if res and len(res) > 0 and res[0] else []
print(f"Total PaddleOCR detected lines on norm_bgr: {len(lines)}")
for item in lines[:25]:
    box = item[0]
    txt, conf = item[1]
    top_y = min(p[1] for p in box)
    left_x = min(p[0] for p in box)
    print(f"y={int(top_y):4d}, x={int(left_x):4d} | conf={conf:.2f} | {txt}")
