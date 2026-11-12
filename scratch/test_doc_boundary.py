import sys
sys.path.insert(0, ".")
import cv2
import numpy as np

def detect_page_contour(image_bgr: np.ndarray):
    h, w = image_bgr.shape[:2]
    total_area = h * w
    scale = 800.0 / max(h, w)
    small = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    sh, sw = small.shape[:2]
    small_area = sh * sw
    
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # Check 1: White paper mask via brightness and low saturation
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v_chan = hsv[:, :, 2]
    s_chan = hsv[:, :, 1]
    
    # White paper is typically high V and lower S compared to colored/dark backgrounds
    # Or Otsu on gray with morphological close
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # If background is bright, check mean of borders
    border_pixels = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
    if np.mean(border_pixels) > 127: # background inverted
        thresh = cv2.bitwise_not(thresh)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 0.25 * small_area and area > max_area:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                best_cnt = approx.reshape(4, 2)
                max_area = area
            elif len(approx) > 4:
                # Minimum area bounding rotated rect
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box)
                if box_area > 0.35 * small_area:
                    best_cnt = box
                    max_area = box_area
                    
    if best_cnt is not None:
        return (best_cnt.astype("float32") / scale), max_area / small_area
    return None, 0.0

test_files = [
    r"data\uploads\25b29d44_WhatsApp Image 03.jpeg",
    r"data\uploads\061d2f34_IMG_4839.JPG 02.jpeg",
    r"data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg",
    r"data\uploads\09b450b9_IMG_5774 01.PNG"
]

for p in test_files:
    img = cv2.imread(p)
    cnt, area_ratio = detect_page_contour(img)
    print(f"{p}: quad found? {cnt is not None} (area ratio: {area_ratio:.2f})")
