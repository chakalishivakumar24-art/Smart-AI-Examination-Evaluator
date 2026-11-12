import os
import cv2
import pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

crop_dir = r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\crop_test"

fields = [
    ("college_name", "--psm 6"),
    ("booklet_serial_number", "-c tessedit_char_whitelist=0123456789 --psm 7"),
    ("test_number", "-c tessedit_char_whitelist=0123456789 --psm 8"),
    ("exam_name", "--psm 6"),
    ("enrollment_number", "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7"),
    ("student_name", "--psm 7"),
    ("section", "-c tessedit_char_whitelist=ABCDE --psm 8"),
    ("exam_date", "-c tessedit_char_whitelist=0123456789./- --psm 7"),
    ("class", "--psm 7"),
    ("course_title", "--psm 7"),
    ("semester", "-c tessedit_char_whitelist=IVXLCDM12345678 --psm 8"),
    ("course_code", "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7"),
    ("branch", "--psm 6"),
    ("additional_sheets", "-c tessedit_char_whitelist=0123456789 --psm 8")
]

print("=== TESTING DIRECT CROP OCR WITH TAILORED PREPROCESSING & WHITELISTS ===")

for field_name, psm_config in fields:
    img_path = os.path.join(crop_dir, f"{field_name}.jpg")
    if not os.path.exists(img_path):
        continue
    
    crop = cv2.imread(img_path)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Add 15px white padding border to prevent edge truncation
    padded = cv2.copyMakeBorder(gray, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    
    # 2.5x Cubic Upscale
    upscaled = cv2.resize(padded, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    # Preprocessing Variants
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(upscaled)
    adaptive = cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    variants = [
        ("padded_raw", upscaled),
        ("clahe", clahe),
        ("adaptive", adaptive),
        ("otsu", otsu)
    ]
    
    print(f"\n--- FIELD: {field_name.upper()} ---")
    for var_name, var_img in variants:
        pil_img = Image.fromarray(var_img)
        try:
            txt = pytesseract.image_to_string(pil_img, config=psm_config).strip()
            data = pytesseract.image_to_data(pil_img, config=psm_config, output_type=pytesseract.Output.DATAFRAME)
            data = data[data.text.notnull() & (data.text.astype(str).str.strip() != '')]
            confs = [float(c) for c in data['conf'] if float(c) > 0]
            avg_conf = float(np.mean(confs)) if confs else 0.0
            print(f"  Variant [{var_name:10s}]: OCR='{txt}' (Conf={avg_conf:.1f}%)")
        except Exception as e:
            print(f"  Variant [{var_name:10s}]: Error={e}")
