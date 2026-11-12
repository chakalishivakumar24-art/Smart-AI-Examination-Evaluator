import sys
sys.path.insert(0, ".")
import cv2
import numpy as np

def order_points(pts):
    # pts: 4x2 array
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # top-left
    rect[2] = pts[np.argmax(s)] # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect

def detect_document_quadrilateral(image_bgr: np.ndarray):
    """
    Detects the 4 corners of the document page in an image.
    Returns ordered (4, 2) float32 coordinates or None if not found.
    """
    h, w = image_bgr.shape[:2]
    orig_area = h * w
    
    # Work on downscaled image for fast and reliable edge detection
    max_dim = 1000
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = image_bgr.copy()
        
    sh, sw = small.shape[:2]
    small_area = sh * sw
    
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try multiple edge detection strategies
    candidate_quads = []
    
    # Strategy 1: Canny edges + dilation
    edges = cv2.Canny(blurred, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 0.30 * small_area:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                candidate_quads.append((area, approx.reshape(4, 2)))
                
    # Strategy 2: Adaptive Threshold / Otsu
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours2, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours2:
        area = cv2.contourArea(c)
        if area > 0.30 * small_area:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                candidate_quads.append((area, approx.reshape(4, 2)))
                
    if not candidate_quads:
        return None
        
    # Sort candidate quads by area descending
    candidate_quads.sort(key=lambda x: x[0], reverse=True)
    best_quad = candidate_quads[0][1]
    
    # Scale coordinates back to original image
    best_quad_orig = best_quad.astype("float32") / scale
    return order_points(best_quad_orig)

def correct_perspective(image_bgr: np.ndarray, target_w: int = 1600, target_h: int = 2200):
    """
    Performs 4-point perspective warp if document boundary is detected.
    Falls back to aspect-ratio preserving crop/resize if not detected.
    """
    h, w = image_bgr.shape[:2]
    quad = detect_document_quadrilateral(image_bgr)
    
    if quad is not None:
        dst = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(quad, dst)
        warped = cv2.warpPerspective(image_bgr, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
        return warped, True
    else:
        # Fallback: simple high-quality resize
        resized = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        return resized, False

# Test on the sample sheets
test_files = [
    r"data\uploads\25b29d44_WhatsApp Image 03.jpeg",
    r"data\uploads\061d2f34_IMG_4839.JPG 02.jpeg",
    r"data\uploads\0db3cc4b_IMG_4867.JPG 01.jpeg",
    r"data\uploads\09b450b9_IMG_5774 01.PNG"
]

for p in test_files:
    img = cv2.imread(p)
    if img is not None:
        warped, detected = correct_perspective(img)
        print(f"File: {p} | Quad detected: {detected} | Warped shape: {warped.shape}")
        out_name = f"scratch\warped_{p.split(chr(92))[-1]}.jpg"
        cv2.imwrite(out_name, warped)
