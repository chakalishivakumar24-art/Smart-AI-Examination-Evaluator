import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional, Any

def preprocess_for_printed_text(crop_bgr: np.ndarray) -> np.ndarray:
    """Grayscale + CLAHE + Adaptive thresholding for printed form labels."""
    if crop_bgr.size == 0:
        return crop_bgr
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    return cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def preprocess_for_handwriting(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Stroke-preserving preprocessing for handwritten text recognition:
    1. Grayscale conversion.
    2. 12px White border padding to prevent edge clipping.
    3. 2.5x Cubic Upscaling so fine pen marks remain continuous.
    4. Bilateral filtering to smooth paper texture/grain while preserving sharp stroke edges.
    5. Adaptive CLAHE contrast enhancement without destructive binary clipping.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr.copy()
    
    # 12px white border padding
    padded = cv2.copyMakeBorder(gray, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    
    # 2.5x Cubic Upscaling
    h, w = padded.shape[:2]
    if h < 100 or w < 200:
        upscaled = cv2.resize(padded, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = padded
        
    # Bilateral smoothing to suppress paper noise without blurring strokes
    denoised = cv2.bilateralFilter(upscaled, d=7, sigmaColor=50, sigmaSpace=50)
    
    # Mild CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return clahe.apply(denoised)

def preprocess_for_table(crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts horizontal and vertical line masks for grid detection."""
    if crop_bgr.size == 0:
        return crop_bgr, crop_bgr, crop_bgr
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    
    th, tw = gray.shape[:2]
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(tw * 0.05)), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(th * 0.05))))

    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
    return gray, h_lines, v_lines

def preprocess_for_digits(crop_bgr: np.ndarray) -> np.ndarray:
    """High contrast binarization tailored for numbers/digits."""
    if crop_bgr.size == 0:
        return crop_bgr
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    upscaled = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(upscaled)
    return cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def convert_to_opencv(image_input) -> np.ndarray:
    """Converts PIL Image or numpy array/bytes to OpenCV BGR image format."""
    if isinstance(image_input, Image.Image):
        from PIL import ImageOps
        transposed = ImageOps.exif_transpose(image_input)
        rgb_array = np.array(transposed.convert('RGB'))
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        return image_input
    elif isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Unable to read image from path: {image_input}")
            return img
        else:
            nparr = np.frombuffer(image_input, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

def rotate_image_angle(image_bgr: np.ndarray, angle: int) -> np.ndarray:
    """Rotates an image by 0, 90, 180, or 270 degrees."""
    if angle == 90:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image_bgr, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_bgr

_TESSERACT_AVAILABLE = True
try:
    import pytesseract
    tess_cmd = os.getenv("TESSERACT_CMD") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tess_cmd):
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
except Exception:
    _TESSERACT_AVAILABLE = False

import time
import os

def ensure_portrait_orientation(image_bgr: np.ndarray) -> np.ndarray:
    """Ensures image is in portrait orientation (height > width). If landscape, rotates 90 deg clockwise."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if w > h:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    return image_bgr

def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 coordinates as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def detect_document_quadrilateral(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """
    OpenCV Document Page Boundary & Quadrilateral Corner Detector.
    Finds the 4 outer corners of the examination sheet against background surfaces.
    Returns (4x2 float32 ordered points, coverage_ratio) or (None, 0.0).
    """
    if image_bgr is None or image_bgr.size == 0:
        return None, 0.0

    h, w = image_bgr.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    small_area = sh * sw

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu thresholding with background polarity check
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    border_pixels = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
    if np.mean(border_pixels) > 127:
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
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box)
                if box_area > 0.35 * small_area and box_area > max_area:
                    best_cnt = box
                    max_area = box_area

    if best_cnt is not None:
        pts_orig = (best_cnt.astype("float32") / scale)
        return order_points(pts_orig), max_area / small_area

    return None, 0.0

def correct_document_perspective(
    image_bgr: np.ndarray,
    target_w: int = 1600,
    target_h: int = 2200
) -> Tuple[np.ndarray, bool]:
    """
    Corrects rotation, perspective skew, and desk margins using 4-point perspective warping.
    Warp output is standardized to the canonical canvas (target_w x target_h).
    Returns (warped_bgr, was_perspective_corrected).
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8), False

    quad, area_ratio = detect_document_quadrilateral(image_bgr)
    dst_pts = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype="float32")

    # Validate quadrilateral geometry:
    # A standard portrait exam page has an aspect ratio (height / width) between 1.20 and 1.65.
    # It must also cover a substantial portion of the vertical extent so the bottom table is not clipped.
    is_valid_quad = False
    if quad is not None and area_ratio > 0.35 and area_ratio < 0.99:
        pts = order_points(quad)
        w_top = np.linalg.norm(pts[1] - pts[0])
        w_bot = np.linalg.norm(pts[2] - pts[3])
        h_left = np.linalg.norm(pts[3] - pts[0])
        h_right = np.linalg.norm(pts[2] - pts[1])
        avg_w = (w_top + w_bot) / 2.0
        avg_h = (h_left + h_right) / 2.0
        aspect = (avg_h / avg_w) if avg_w > 0 else 0.0
        h_img = image_bgr.shape[0]
        
        # Ensure aspect ratio is plausible for an exam page and doesn't chop off the bottom table
        if 1.15 <= aspect <= 1.65 and (avg_h / h_img) >= 0.75:
            is_valid_quad = True
            M = cv2.getPerspectiveTransform(pts, dst_pts)
            warped = cv2.warpPerspective(image_bgr, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
            return warped, True

    # High quality proportional canvas resize fallback
    resized = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return resized, False

def remove_shadows_and_enhance(image_bgr: np.ndarray) -> np.ndarray:
    """
    OpenCV shadow compensation and contrast normalization:
    Checks illumination variance: only applies background division when noticeable shadow gradients exist,
    preserving thin handwritten pen strokes and digits on flatbed scans.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    corners = [
        float(np.mean(gray[:h//4, :w//4])),
        float(np.mean(gray[:h//4, 3*w//4:])),
        float(np.mean(gray[3*h//4:, :w//4])),
        float(np.mean(gray[3*h//4:, 3*w//4:]))
    ]
    illumination_spread = max(corners) - min(corners)

    # If illumination is fairly uniform (e.g. flatbed scan or good lighting),
    # preserve original ink contrast without destructive background division
    if illumination_spread < 65:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

    channels = cv2.split(image_bgr)
    norm_channels = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))

    for ch in channels:
        dilated = cv2.dilate(ch, kernel)
        bg = cv2.medianBlur(dilated, 31)
        diff = 255 - cv2.absdiff(ch, bg)
        norm_ch = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        norm_channels.append(norm_ch)

    return cv2.merge(norm_channels)

def is_valid_crop(crop_bgr: np.ndarray, min_w: int = 10, min_h: int = 8) -> bool:
    """Rejects empty, invalid, or incorrectly sized image crops."""
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    h, w = crop_bgr.shape[:2]
    if w < min_w or h < min_h:
        return False
    return True

def find_best_orthogonal_orientation(image_bgr: np.ndarray, is_first_page: bool = True) -> Tuple[np.ndarray, int]:
    """
    Tests orthogonal page orientations (0, 90, 180, 270 degrees) using deep learning
    text recognition, keyword scoring, and document layout heuristics.
    Returns (rotated_upright_image, detected_angle).
    """
    from ocr.orientation import find_best_orthogonal_orientation as _find_best_orientation
    return _find_best_orientation(image_bgr, is_first_page=is_first_page)

from ocr.orientation import (
    detect_orientation_angle,
    detect_and_rotate_image,
    transform_bbox,
    inverse_transform_bbox
)





def detect_form_field_regions(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    OpenCV Layout & Form Field Bounding Box Region Segmenter:
    Uses morphological line kernels to detect rectangular form fields, underline strokes,
    and bounding boxes on Page 1 for isolated handwritten text recognition.
    Returns list of bounding boxes (x, y, w, h).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)

    table_grid = cv2.add(horiz_lines, vert_lines)
    contours, _ = cv2.findContours(table_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h_img, w_img = image_bgr.shape[:2]
    min_w, min_h = int(w_img * 0.05), int(h_img * 0.015)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_w and h >= min_h and w < int(w_img * 0.95):
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: (b[1] // 30, b[0]))
    return boxes

def preprocess_image(image_input, auto_rotate: bool = True, enhance_contrast: bool = True) -> Image.Image:
    """
    Standardized Preprocessing API:
    Preprocesses an image for OCR and returns the processed PIL Image.
    """
    _, pil_image, _ = preprocess_image_for_ocr(image_input, auto_rotate=auto_rotate, enhance_contrast=enhance_contrast)
    return pil_image


def preprocess_image_for_ocr(image_input, auto_rotate: bool = True, enhance_contrast: bool = True) -> Tuple[np.ndarray, Image.Image, int]:
    """
    Applies computer vision preprocessing pipeline for printed and handwritten OCR:
    1. OpenCV Image Conversion & High Resolution Upscaling (300 DPI scale)
    2. 4-Angle Orthogonal Auto-Rotation Detection (0, 90, 180, 270 deg)
    3. Grayscale Conversion & Bilateral Noise Denoising
    4. CLAHE Contrast Enhancement
    5. Adaptive Thresholding
    Returns processed OpenCV image, PIL Image, and detected orientation angle (3 values).
    """
    img = convert_to_opencv(image_input)
    detected_angle = 0

    height, width = img.shape[:2]
    if width < 1800:
        scale = 1800.0 / width
        img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)

    if auto_rotate:
        img, detected_angle = find_best_orthogonal_orientation(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
    else:
        enhanced = denoised

    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    pil_image = Image.fromarray(thresh)
    return thresh, pil_image, detected_angle

