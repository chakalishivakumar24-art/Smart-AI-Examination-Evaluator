import sys
sys.path.insert(0, ".")
import os
import cv2
from ocr.paddle_ocr import get_paddleocr_engine

engine = get_paddleocr_engine()
sample_files = [
    r"data\test_normalized_page.jpg",
    r"data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg",
    r"data\uploads\061d2f34_IMG_4839.JPG 02.jpeg",
    r"data\uploads\09b450b9_IMG_5774 01.PNG",
    r"data\uploads\25b29d44_WhatsApp Image 03.jpeg"
]

out_path = r"scratch\paddle_inspect_output.txt"
with open(out_path, "w", encoding="utf-8") as out_f:
    for p in sample_files:
        if not os.path.exists(p):
            out_f.write(f"Skipping {p}: not found\n")
            out_f.flush()
            continue
        img = cv2.imread(p)
        if img is None:
            out_f.write(f"Skipping {p}: failed to load\n")
            out_f.flush()
            continue
        h, w = img.shape[:2]
        out_f.write(f"\n========================================================\n")
        out_f.write(f"FILE: {p} (shape: {img.shape})\n")
        out_f.flush()
        
        # If very high resolution, resize to height 2200 for inspection
        if h > 2500:
            scale = 2200.0 / h
            img_proc = cv2.resize(img, (int(w * scale), 2200), interpolation=cv2.INTER_AREA)
        else:
            img_proc = img

        text, conf, detections = engine.read_text_with_boxes(img_proc, enhance_contrast=False)
        out_f.write(f"Detections count: {len(detections)}, avg conf: {conf:.1f}%\n")
        out_f.write(f"--------------------------------------------------------\n")
        for d in detections:
            rx, ry, rw, rh = d["rect"]
            out_f.write(f"  [{rx:4d}, {ry:4d}, {rw:4d}, {rh:4d}] ({d['confidence']:5.1f}%): {d['text']}\n")
        out_f.flush()
print("Done inspection!")
