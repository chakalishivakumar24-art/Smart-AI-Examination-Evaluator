import os
import cv2
import re
import pytesseract
import numpy as np
from PIL import Image
from template_config import BOOKLET_TEMPLATE, NORMALIZED_WIDTH, NORMALIZED_HEIGHT

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Test sample images
sample_images = [
    r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\test_normalized_page.jpg",
    r"c:\Users\vamshi\Desktop\SMART AI EXAMNITATION EVALUTOR\data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg"
]

PSM_CONFIGS = {
    "booklet_serial_number": "-c tessedit_char_whitelist=0123456789 --psm 7",
    "test_number": "-c tessedit_char_whitelist=0123456789 --psm 8",
    "additional_sheets": "-c tessedit_char_whitelist=0123456789 --psm 8",
    "section": "-c tessedit_char_whitelist=ABCDE --psm 8",
    "semester": "-c tessedit_char_whitelist=IVXLCDM12345678 --psm 8",
    "course_code": "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7",
    "enrollment_number": "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7",
    "exam_date": "-c tessedit_char_whitelist=0123456789./- --psm 7",
    "student_name": "--psm 7",
    "college_name": "--psm 6",
    "exam_name": "--psm 6",
    "course_title": "--psm 7",
    "branch": "--psm 6",
    "department": "--psm 6",
    "class": "--psm 7",
    "instructor_name": "--psm 7",
    "examiner_remarks": "--psm 6",
    "recheck_request": "--psm 6"
}

def clean_field_ocr(raw: str, field_name: str) -> str:
    if not raw or not raw.strip():
        return ""
    
    val = raw.strip()
    val = re.sub(r'[\r\n\t]+', ' ', val)
    val = re.sub(r"^(?:NO|NUM|NUMBER|ENROLLMENT|REGISTRATION|CODE|SEM|DATE|NAME|SECTION)[:\s\-=]*", "", val, flags=re.IGNORECASE).strip()
    val = re.sub(r"^[\:\-=\s\(\)\[\]\{\}\"\']+", "", val)
    val = re.sub(r"[\:\-=\s\(\)\[\]\{\}\"\']+$", "", val).strip()
    
    f = field_name.lower()
    
    if f in ["booklet_serial_number", "test_number", "additional_sheets"]:
        digits = re.sub(r"[^\d]", "", val)
        return digits
    
    elif f == "enrollment_number":
        # Strips noise prefixes like 'NO' or 'N'
        val_clean = re.sub(r"^(?:NO|NUM|N)\s*", "", val, flags=re.IGNORECASE).strip()
        val_clean = re.sub(r"[^\w]", "", val_clean).upper()
        # Fix common OCR confusion e.g. 'O' -> '0' in numeric zones
        match = re.search(r"([A-Z0-9]{6,22})", val_clean)
        return match.group(1) if match else val_clean
        
    elif f == "section":
        match = re.search(r"([A-E])", val, re.IGNORECASE)
        return match.group(1).upper() if match else ""
        
    elif f == "semester":
        match = re.search(r"(I{1,3}|IV|V|VI{0,2}|VIII|[1-8])", val, re.IGNORECASE)
        if match:
            s_val = match.group(1).upper()
            num_map = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI", "7": "VII", "8": "VIII"}
            return num_map.get(s_val, s_val)
        return ""
        
    elif f in ["course_code", "subject_code"]:
        match = re.search(r"([A-Z]{2,4}\s*\d{3,4})", val, re.IGNORECASE)
        if match:
            return re.sub(r"\s+", "", match.group(1)).upper()
        clean = re.sub(r"[^\w]", "", val).upper()
        return clean
        
    elif f == "exam_date":
        match = re.search(r"(\d{1,2})[\-/\.](\d{1,2})[\-/\.](\d{2,4})", val)
        if match:
            d, m, y = match.group(1).zfill(2), match.group(2).zfill(2), match.group(3)
            if len(y) == 2:
                y = "20" + y
            return f"{d}/{m}/{y}"
        return val
        
    elif f == "student_name":
        clean = re.sub(r"[^\w\s\.]", "", val).strip()
        clean = re.sub(r"^[T\W_]+", "", clean).strip()
        return clean

    return val

print("=== TESTING FIXED TEMPLATE ROIS WITH PADDING & MULTI-PASS PREPROCESSING ===")

for img_path in sample_images:
    if not os.path.exists(img_path):
        continue
    print(f"\n=======================================================")
    print(f"IMAGE: {os.path.basename(img_path)}")
    print(f"=======================================================")
    
    bgr = cv2.imread(img_path)
    h, w = bgr.shape[:2]
    if w != NORMALIZED_WIDTH or h != NORMALIZED_HEIGHT:
        bgr = cv2.resize(bgr, (NORMALIZED_WIDTH, NORMALIZED_HEIGHT), interpolation=cv2.INTER_CUBIC)
        
    for field_name, (x, y, bw, bh) in BOOKLET_TEMPLATE.items():
        if field_name in ["marks_table", "student_signature", "invigilator_signature", "examiner_signature"]:
            continue
            
        crop_bgr = bgr[y:y+bh, x:x+bw]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        
        # 15px white padding
        padded = cv2.copyMakeBorder(gray, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
        # 2.5x Cubic upscale
        upscaled = cv2.resize(padded, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(upscaled)
        adaptive = cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        variants = [("padded", upscaled), ("clahe", clahe), ("adaptive", adaptive), ("otsu", otsu)]
        psm_cfg = PSM_CONFIGS.get(field_name, "--psm 6")
        
        best_cleaned = ""
        best_conf = 0.0
        best_var = ""
        
        for var_name, var_img in variants:
            pil_img = Image.fromarray(var_img)
            try:
                data = pytesseract.image_to_data(pil_img, config=psm_cfg, output_type=pytesseract.Output.DATAFRAME)
                data = data[data.text.notnull() & (data.text.astype(str).str.strip() != '')]
                if data.empty:
                    continue
                raw_txt = " ".join(data['text'].astype(str)).strip()
                confs = [float(c) for c in data['conf'] if float(c) > 0]
                avg_c = float(np.mean(confs)) if confs else 0.0
                
                cleaned = clean_field_ocr(raw_txt, field_name)
                if cleaned and (avg_c > best_conf or not best_cleaned):
                    best_cleaned = cleaned
                    best_conf = avg_c
                    best_var = var_name
            except Exception:
                continue
                
        status = "OK" if best_conf >= 55.0 and best_cleaned else ("NEEDS_REVIEW" if best_cleaned else "NOT_DETECTED")
        print(f"[FIELD] {field_name:22s} = '{best_cleaned}' (Conf: {best_conf:.1f}%, Var: {best_var}, Status: {status})")
