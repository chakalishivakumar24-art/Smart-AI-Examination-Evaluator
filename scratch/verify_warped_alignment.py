import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from scratch.test_perspective import order_points
from scratch.test_doc_boundary import detect_page_contour
from ocr.paddle_ocr import get_paddleocr_engine

engine = get_paddleocr_engine()
target_w, target_h = 1600, 2200
dst_pts = np.array([
    [0, 0],
    [target_w - 1, 0],
    [target_w - 1, target_h - 1],
    [0, target_h - 1]
], dtype="float32")

test_files = [
    r"data\uploads\09b450b9_IMG_5774 01.PNG",
    r"data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg",
    r"data\uploads\061d2f34_IMG_4839.JPG 02.jpeg",
    r"data\uploads\25b29d44_WhatsApp Image 03.jpeg"
]

for p in test_files:
    img = cv2.imread(p)
    cnt, area_ratio = detect_page_contour(img)
    if cnt is not None and area_ratio > 0.40 and area_ratio < 0.99:
        ordered = order_points(cnt)
        M = cv2.getPerspectiveTransform(ordered, dst_pts)
        warped = cv2.warpPerspective(img, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
    else:
        warped = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
    cv2.imwrite(f"scratch\warped_{p.split(chr(92))[-1]}.jpg", warped)
    
    # Run PaddleOCR on the warped canonical image
    text, conf, detections = engine.read_text_with_boxes(warped, enhance_contrast=False)
    print(f"\n=== WARPED: {p} ===")
    for d in detections:
        txt = d["text"].lower()
        if any(k in txt for k in ["booklet", "serial", "icfai", "enroll", "name:", "section", "date", "class", "course", "semester", "grand", "total"]):
            print(f"  {d['rect']}: {d['text']} ({d['confidence']}%)")
