import cv2
import os
from template_config import BOOKLET_TEMPLATE, NORMALIZED_WIDTH, NORMALIZED_HEIGHT

img_path = r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\test_normalized_page.jpg"
if not os.path.exists(img_path):
    print("No sample image found!")
    exit(1)

bgr = cv2.imread(img_path)
h, w = bgr.shape[:2]
if w != NORMALIZED_WIDTH or h != NORMALIZED_HEIGHT:
    bgr = cv2.resize(bgr, (NORMALIZED_WIDTH, NORMALIZED_HEIGHT), interpolation=cv2.INTER_CUBIC)

canvas = bgr.copy()

colors = {
    "student_name": (0, 255, 0),
    "enrollment_number": (255, 0, 0),
    "course_title": (0, 0, 255),
    "course_code": (255, 255, 0),
    "semester": (255, 0, 255),
    "section": (0, 255, 255),
    "exam_date": (128, 255, 0),
    "booklet_serial_number": (0, 128, 255)
}

for name, (x, y, bw, bh) in BOOKLET_TEMPLATE.items():
    color = colors.get(name, (200, 200, 200))
    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)
    cv2.putText(canvas, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

out_path = r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\page1_roi_debug.jpg"
cv2.imwrite(out_path, canvas)
print(f"Saved ROI debug image to {out_path}")
