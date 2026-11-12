import os
import sys
sys.path.insert(0, ".")
import cv2
from PIL import Image, ImageOps
import pytesseract
from ocr.preprocessing import convert_to_opencv, rotate_image_angle
from ocr.paddle_ocr import get_paddleocr_engine

img_path = r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg"
pil_raw = Image.open(img_path)
pil_trans = ImageOps.exif_transpose(pil_raw)

print(f"pil_raw size: {pil_raw.size}")
print(f"pil_trans size: {pil_trans.size}")

# Check Tesseract on raw vs transposed vs rotated
for name, p_img in [("raw_0", pil_raw), ("transposed", pil_trans)]:
    txt = pytesseract.image_to_string(p_img.resize((800, int(800 * p_img.height / p_img.width))), config='--psm 3')
    found = [w for w in ["icfai", "examination", "comprehensive", "student", "roll", "enrollment", "bhavana", "name", "marks"] if w in txt.lower()]
    print(f"[{name}] matches: {found}")
    print(f"[{name}] sample text: {repr(txt[:100])}")

# Let's also check raw rotated 0, 90, 180, 270
raw_bgr = convert_to_opencv(pil_raw)
for ang in [0, 90, 180, 270]:
    rot = rotate_image_angle(raw_bgr, ang)
    # Check if height > width
    h, w = rot.shape[:2]
    # Let's test with paddleocr
    p_img = Image.fromarray(cv2.cvtColor(cv2.resize(rot, (800, int(800 * h / w))), cv2.COLOR_BGR2RGB))
    txt = pytesseract.image_to_string(p_img, config='--psm 3')
    found = [w for w in ["icfai", "examination", "comprehensive", "student", "roll", "enrollment", "bhavana", "name", "marks"] if w in txt.lower()]
    print(f"[raw rot {ang}° (shape={w}x{h})] matches: {found}")
