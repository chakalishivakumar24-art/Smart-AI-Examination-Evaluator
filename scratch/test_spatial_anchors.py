import sys
sys.path.insert(0, ".")
import cv2
import re
import numpy as np
from ocr.paddle_ocr import get_paddleocr_engine

engine = get_paddleocr_engine()

def extract_fields_with_spatial_anchors(img_bgr: np.ndarray):
    """
    Extracts all 12 metadata fields and marks using dynamic spatial label anchoring:
    1. PaddleOCR locates the printed label anchor bounding boxes on the page.
    2. The field crop boundaries are calculated directly relative to each anchor,
       strictly excluding the printed label and neighboring fields.
    3. Each isolated field crop is recognized with PaddleOCR.
    """
    h, w = img_bgr.shape[:2]
    text, conf, detections = engine.read_text_with_boxes(img_bgr, enhance_contrast=False)
    
    # Map label anchors to their bounding rectangles
    anchors = {}
    for d in detections:
        raw_t = d["text"].strip()
        t_low = raw_t.lower()
        rx, ry, rw, rh = d["rect"]
        
        if "booklet" in t_low and "serial" in t_low:
            anchors["booklet_serial"] = (rx, ry, rw, rh, raw_t)
        elif "test" in t_low and "comprehensive" in t_low:
            anchors["test_exam_line"] = (rx, ry, rw, rh, raw_t)
        elif "enroll" in t_low:
            anchors["enrollment"] = (rx, ry, rw, rh, raw_t)
        elif "instructor" in t_low:
            anchors["instructor"] = (rx, ry, rw, rh, raw_t)
        elif t_low.startswith("name:") or t_low == "name:" or "name:" in t_low:
            anchors["name"] = (rx, ry, rw, rh, raw_t)
        elif "section" in t_low:
            anchors["section"] = (rx, ry, rw, rh, raw_t)
        elif "date:" in t_low or t_low.startswith("date"):
            anchors["date"] = (rx, ry, rw, rh, raw_t)
        elif "class" in t_low and "of" in t_low:
            anchors["class"] = (rx, ry, rw, rh, raw_t)
        elif "course" in t_low and "title" in t_low:
            anchors["course_title"] = (rx, ry, rw, rh, raw_t)
        elif "semester" in t_low:
            anchors["semester"] = (rx, ry, rw, rh, raw_t)
        elif "course" in t_low and "code" in t_low:
            anchors["course_code"] = (rx, ry, rw, rh, raw_t)
        elif "additional" in t_low and "sheet" in t_low:
            anchors["additional_sheets"] = (rx, ry, rw, rh, raw_t)
        elif "grand" in t_low and "total" in t_low:
            anchors["grand_total"] = (rx, ry, rw, rh, raw_t)
            
    field_crops = {}
    field_results = {}
    
    # 1. Booklet Serial Number: to the right of Booklet Serial No:
    if "booklet_serial" in anchors:
        ax, ay, aw, ah, _ = anchors["booklet_serial"]
        # Look for numbers to the right
        cx1 = ax + aw + 5
        cy1 = max(0, ay - 10)
        cx2 = min(w, ax + aw + 400)
        cy2 = min(h, ay + ah + 20)
        crop = img_bgr[cy1:cy2, cx1:cx2]
        field_crops["booklet_serial_number"] = crop
        txt, cnf = engine.read_field_crop(crop)
        clean = re.sub(r"[^\d]", "", txt)
        field_results["booklet_serial_number"] = (clean, cnf, (cx1, cy1, cx2-cx1, cy2-cy1))

    # 2. Test Number
    if "test_exam_line" in anchors:
        ax, ay, aw, ah, raw_t = anchors["test_exam_line"]
        # Often "Test 1 / Comprehensive Examination" or "Test_Z/Comprehensive Examination"
        m_t = re.search(r"Test[_\s]*([0-9A-Z]+)", raw_t, re.IGNORECASE)
        if m_t:
            field_results["test_number"] = (m_t.group(1).replace("Z", "2"), 90.0, (ax, ay, aw, ah))
        else:
            field_results["test_number"] = ("1", 75.0, (ax, ay, aw, ah))

    # 3. Enrollment Number: in the cell to the right of Enrollment No:
    if "enrollment" in anchors:
        ax, ay, aw, ah, raw_t = anchors["enrollment"]
        # Check if enrollment text already includes the value
        m_e = re.search(r"(2[34][A-Z0-9]{8,14})", raw_t, re.IGNORECASE)
        if m_e:
            field_results["enrollment_number"] = (m_e.group(1).upper(), 95.0, (ax, ay, aw, ah))
        else:
            cx1 = ax + aw + 5
            cy1 = max(0, ay - 8)
            cx2 = min(w, ax + aw + 550)
            cy2 = min(h, ay + ah + 25)
            crop = img_bgr[cy1:cy2, cx1:cx2]
            field_crops["enrollment_number"] = crop
            txt, cnf = engine.read_field_crop(crop)
            clean = re.sub(r"[^\w]", "", txt).upper()
            field_results["enrollment_number"] = (clean, cnf, (cx1, cy1, cx2-cx1, cy2-cy1))

    # 4. Student Name: between Name: and Section:
    if "name" in anchors:
        ax, ay, aw, ah, raw_t = anchors["name"]
        m_n = re.search(r"Name\s*[:\.\-]*\s*([A-Za-z\s\.]+)", raw_t, re.IGNORECASE)
        if m_n and len(m_n.group(1).strip()) > 3:
            field_results["student_name"] = (m_n.group(1).strip(), 92.0, (ax, ay, aw, ah))
        else:
            cx1 = ax + aw + 8
            cy1 = max(0, ay - 6)
            cx2 = min(w, ax + aw + 480)
            if "section" in anchors:
                cx2 = min(cx2, anchors["section"][0] - 10)
            cy2 = min(h, ay + ah + 20)
            crop = img_bgr[cy1:cy2, cx1:cx2]
            field_crops["student_name"] = crop
            txt, cnf = engine.read_field_crop(crop)
            clean = re.sub(r"^(?:Name|NAME)[:\s\-=]*", "", txt).strip()
            field_results["student_name"] = (clean, cnf, (cx1, cy1, cx2-cx1, cy2-cy1))

    # 5. Section: between Section: and Date:
    if "section" in anchors:
        ax, ay, aw, ah, raw_t = anchors["section"]
        m_s = re.search(r"Section\s*[:\.\-]*\s*([A-Za-z0-9])", raw_t, re.IGNORECASE)
        if m_s:
            field_results["section"] = (m_s.group(1).upper(), 92.0, (ax, ay, aw, ah))
        else:
            cx1 = ax + aw + 5
            cy1 = max(0, ay - 6)
            cx2 = min(w, ax + aw + 220)
            if "date" in anchors:
                cx2 = min(cx2, anchors["date"][0] - 10)
            cy2 = min(h, ay + ah + 20)
            crop = img_bgr[cy1:cy2, cx1:cx2]
            field_crops["section"] = crop
            txt, cnf = engine.read_field_crop(crop)
            clean = re.sub(r"^(?:Section|Sec)[:\s\-=]*", "", txt).strip()
            field_results["section"] = (clean[:1].upper() if clean else "", cnf, (cx1, cy1, cx2-cx1, cy2-cy1))

    # 6. Date: to the right of Date:
    if "date" in anchors:
        ax, ay, aw, ah, raw_t = anchors["date"]
        m_d = re.search(r"Date\s*[:\.\-]*\s*(\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4})", raw_t, re.IGNORECASE)
        if m_d:
            field_results["exam_date"] = (m_d.group(1), 92.0, (ax, ay, aw, ah))
        else:
            cx1 = ax + aw + 5
            cy1 = max(0, ay - 6)
            cx2 = min(w, ax + aw + 350)
            cy2 = min(h, ay + ah + 20)
            crop = img_bgr[cy1:cy2, cx1:cx2]
            field_crops["exam_date"] = crop
            txt, cnf = engine.read_field_crop(crop)
            clean = re.sub(r"^(?:Date)[:\s\-=]*", "", txt).strip()
            field_results["exam_date"] = (clean, cnf, (cx1, cy1, cx2-cx1, cy2-cy1))

    # 7. Class: between Class of: and Course Title:
    if "class" in anchors:
        ax, ay, aw, ah, raw_t = anchors["class"]
        m_c = re.search(r"Class\s*of\s*[:\.\-]*\s*([A-Za-z0-9\s\-–—]+)", raw_t, re.IGNORECASE)
        if m_c and len(m_c.group(1).strip()) > 2:
            field_results["class"] = (m_c.group(1).strip(), 90.0, (ax, ay, aw, ah))
        else:
            cx1 = ax + aw + 8
            cy1 = max(0, ay - 6)
            cx2 = min(w, ax + aw + 450)
            if "course_title" in anchors:
                cx2 = min(cx2, anchors["course_title"][0] - 10)
            cy2 = min(h, ay + ah + 20)
            crop = img_bgr[cy1:cy2, cx1:cx2]
            field_crops["class"] = crop
            txt, cnf = engine.read_field_crop(crop)
            clean = re.sub(r"^(?:Class\s*of)[:\s\-=]*", "", txt).strip()
            field_results["class"] = (clean, cnf, (cx1, cy1, cx2-cx1, cy2-cy1))

    # 8. Course Title: to the right of Course Title:
    if "course_title" in anchors:
        ax, ay, aw, ah, raw_t = anchors["course_title"]
        m_ct = re.search(r"Course\s*Title\s*[:\.\-]*\s*([A-Za-z\s&]+)", raw_t, re.IGNORECASE)
        if m_ct and len(m_ct.group(1).strip()) > 2:
            field_results["course_title"] = (m_ct.group(1).strip(), 90.0, (ax, ay, aw, ah))
        else:
            cx1 = ax + aw + 8
            cy1 = max(0, ay - 6)
            cx2 = min(w, ax + aw + 650)
            cy2 = min(h, ay + ah + 25)
            crop = img_bgr[cy1:cy2, cx1:cx2]
            field_crops["course_title"] = crop
            txt, cnf = engine.read_field_crop(crop)
            clean = re.sub(r"^(?:Course\s*Title)[:\s\-=]*", "", txt).strip()
            field_results["course_title"] = (clean, cnf, (cx1, cy1, cx2-cx1, cy2-cy1))

    # 9. Semester: to the right of Semester:
    if "semester" in anchors:
        ax, ay, aw, ah, raw_t = anchors["semester"]
        m_sm = re.search(r"Semester\s*[:\.\-]*\s*([IVXLCDM0-9]+)", raw_t, re.IGNORECASE)
        if m_sm:
            field_results["semester"] = (m_sm.group(1).upper(), 92.0, (ax, ay, aw, ah))
        else:
            cx1 = ax + aw + 5
            cy1 = max(0, ay - 5)
            cx2 = min(w, ax + aw + 250)
            cy2 = min(h, ay + ah + 20)
            crop = img_bgr[cy1:cy2, cx1:cx2]
            field_crops["semester"] = crop
            txt, cnf = engine.read_field_crop(crop)
            clean = re.sub(r"^(?:Semester)[:\s\-=]*", "", txt).strip().upper()
            field_results["semester"] = (clean, cnf, (cx1, cy1, cx2-cx1, cy2-cy1))

    # 10. Course Code: to the right of Course Code:
    if "course_code" in anchors:
        ax, ay, aw, ah, raw_t = anchors["course_code"]
        m_cc = re.search(r"Course\s*Code\s*[:\.\-]*\s*([A-Za-z0-9]+)", raw_t, re.IGNORECASE)
        if m_cc:
            field_results["course_code"] = (m_cc.group(1).upper(), 92.0, (ax, ay, aw, ah))
        else:
            cx1 = ax + aw + 5
            cy1 = max(0, ay - 5)
            cx2 = min(w, ax + aw + 300)
            cy2 = min(h, ay + ah + 20)
            crop = img_bgr[cy1:cy2, cx1:cx2]
            field_crops["course_code"] = crop
            txt, cnf = engine.read_field_crop(crop)
            clean = re.sub(r"^(?:Course\s*Code)[:\s\-=]*", "", txt).strip().upper()
            field_results["course_code"] = (clean, cnf, (cx1, cy1, cx2-cx1, cy2-cy1))

    # 11. Instructor Name:
    if "instructor" in anchors:
        ax, ay, aw, ah, raw_t = anchors["instructor"]
        m_in = re.search(r"Instructor(?:'s)?\s*Name\s*[:\.\-]*\s*([A-Za-z\s\.]+)", raw_t, re.IGNORECASE)
        if m_in and len(m_in.group(1).strip()) > 2:
            field_results["instructor_name"] = (m_in.group(1).strip(), 90.0, (ax, ay, aw, ah))
        else:
            cx1 = ax + aw + 5
            cy1 = max(0, ay - 5)
            cx2 = min(w, ax + aw + 450)
            cy2 = min(h, ay + ah + 25)
            crop = img_bgr[cy1:cy2, cx1:cx2]
            txt, cnf = engine.read_field_crop(crop)
            clean = re.sub(r"^(?:Instructor(?:'s)?\s*Name)[:\s\-=]*", "", txt).strip()
            field_results["instructor_name"] = (clean, cnf, (cx1, cy1, cx2-cx1, cy2-cy1))
            
    # 12. Grand Total
    if "grand_total" in anchors:
        ax, ay, aw, ah, raw_t = anchors["grand_total"]
        # Look around row 10 / grand total area for circled or written marks
        cx1 = max(0, ax - 100)
        cy1 = max(0, ay - 120)
        cx2 = min(w, ax + 500)
        cy2 = min(h, ay + 100)
        crop = img_bgr[cy1:cy2, cx1:cx2]
        field_crops["grand_total"] = crop
        gt_t, gt_c = engine.read_field_crop(crop)
        field_results["grand_total"] = (gt_t, gt_c, (cx1, cy1, cx2-cx1, cy2-cy1))

    return field_results, field_crops

# Test on the 4 sample files
test_files = [
    (r"scratch\warped_09b450b9_IMG_5774 01.PNG.jpg", "IMG_5774"),
    (r"scratch\warped_0db3cc4b_IMG_4867.JPG 01.jpeg.jpg", "IMG_4867"),
    (r"scratch\warped_061d2f34_IMG_4839.JPG 02.jpeg.jpg", "IMG_4839"),
    (r"scratch\warped_25b29d44_WhatsApp Image 03.jpeg.jpg", "WhatsApp_03")
]

for p, lbl in test_files:
    img = cv2.imread(p)
    if img is None:
        continue
    res, crops = extract_fields_with_spatial_anchors(img)
    print(f"\n========================================================")
    print(f"RESULTS FOR {lbl}:")
    print(f"========================================================")
    for k, (v, c, bbox) in res.items():
        print(f"  {k:22s}: '{v}' (conf={c:.1f}%, bbox={bbox})")
