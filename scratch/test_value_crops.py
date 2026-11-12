import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from ocr.paddle_ocr import get_paddleocr_engine

engine = get_paddleocr_engine()

# Clean value-only ROIs on 1600 x 2200 canvas (x, y, w, h)
VALUE_ROIS = {
    "booklet_serial_number": (1100, 70, 420, 105),
    "test_number": (370, 310, 180, 80),
    "enrollment_number": (1030, 355, 500, 85),
    "instructor_name": (1030, 440, 500, 85),
    "student_name": (190, 530, 420, 80),
    "section": (745, 530, 270, 80),
    "exam_date": (1145, 530, 385, 80),
    "class": (220, 610, 400, 80),
    "course_title": (850, 605, 680, 85),
    "semester": (220, 690, 400, 70),
    "course_code": (260, 760, 360, 70),
    "additional_sheets": (380, 830, 240, 70),
    "student_signature": (970, 690, 560, 70),
    "invigilator_signature": (980, 760, 550, 70),
    "grand_total": (550, 1940, 200, 120)
}

test_warped_files = [
    (r"scratch\warped_09b450b9_IMG_5774 01.PNG.jpg", "IMG_5774"),
    (r"scratch\warped_0db3cc4b_IMG_4867.JPG 01.jpeg.jpg", "IMG_4867"),
    (r"scratch\warped_061d2f34_IMG_4839.JPG 02.jpeg.jpg", "IMG_4839"),
    (r"scratch\warped_25b29d44_WhatsApp Image 03.jpeg.jpg", "WhatsApp_03")
]

for img_p, label in test_warped_files:
    img = cv2.imread(img_p)
    if img is None:
        continue
    print(f"\n========================================================")
    print(f"TESTING VALUE-ONLY CROPS ON: {label}")
    print(f"========================================================")
    for fname, (x, y, w, h) in VALUE_ROIS.items():
        crop = img[y:y+h, x:x+w]
        txt, conf = engine.read_field_crop(crop)
        # Check dark pixels
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark = np.sum(gray < 170)
        cv2.imwrite(f"scratch\crop_{label}_{fname}.jpg", crop)
        print(f"  {fname:22s}: OCR='{txt:20s}' (conf={conf:5.1f}%, dark={dark:5d})")
