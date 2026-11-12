import os
import cv2
import numpy as np
from PIL import Image
from template_config import BOOKLET_TEMPLATE, NORMALIZED_WIDTH, NORMALIZED_HEIGHT

# Load sample image
sample_paths = [
    r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\test_normalized_page.jpg",
    r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg"
]

img_path = None
for p in sample_paths:
    if os.path.exists(p):
        img_path = p
        break

if not img_path:
    print("No sample image found!")
    exit(1)

print(f"Loading sample image: {img_path}")
bgr = cv2.imread(img_path)
h, w = bgr.shape[:2]
if w != NORMALIZED_WIDTH or h != NORMALIZED_HEIGHT:
    bgr = cv2.resize(bgr, (NORMALIZED_WIDTH, NORMALIZED_HEIGHT), interpolation=cv2.INTER_CUBIC)

out_dir = r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\crop_test"
os.makedirs(out_dir, exist_ok=True)

for field_name, (x, y, bw, bh) in BOOKLET_TEMPLATE.items():
    crop = bgr[y:y+bh, x:x+bw]
    out_path = os.path.join(out_dir, f"{field_name}.jpg")
    cv2.imwrite(out_path, crop)
    print(f"Saved {field_name:22s}: bbox=({x}, {y}, {bw}, {bh}), size={crop.shape[1]}x{crop.shape[0]}")

print("Saved all crops to data/crop_test!")
