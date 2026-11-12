import sys
import os
sys.path.insert(0, os.path.abspath("."))
import cv2
from ocr.paddle_ocr import get_paddleocr_engine

engine = get_paddleocr_engine()
image_paths = [
    'data/uploads/79191a30_WhatsApp Image 006.jpeg',
    'data/uploads/3c7d95bb_WhatsApp Image 005.jpeg',
    'data/uploads/e65f4245_IMG_5827.JPG 002.jpeg',
    'data/uploads/9fb5d70e_IMG_5829.JPG 003.jpeg',
    'data/uploads/4ddfea8c_Image 004.jpeg'
]

for p in image_paths:
    if not os.path.exists(p):
        continue
    img = cv2.imread(p)
    print(f"==================================================")
    print(f"FILE: {p} (shape: {img.shape if img is not None else None})")
    if img is not None:
        txt, cnf, boxes = engine.read_text_with_boxes(img, enhance_contrast=False)
        print(f"Total detections: {len(boxes)}, conf: {cnf:.1f}%")
        for b in boxes:
            print(f"  {b['rect']} ({b['confidence']:.1f}%): {b['text']}")
