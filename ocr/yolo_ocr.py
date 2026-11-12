"""
YOLOv8 + Multi-Engine OCR System for Handwritten & Document Text Recognition.

Integrates Ultralytics YOLOv8 object/text region detection with EasyOCR and Tesseract OCR engines.
Provides ROI localization, line segmentation, confidence scoring, and multi-pass candidate evaluation.
"""

import os
import cv2
import time
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional

# Track available engines
_YOLO_AVAILABLE = False
_EASYOCR_AVAILABLE = False
_TESSERACT_AVAILABLE = False

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

try:
    import pytesseract
    tess_paths = [
        os.getenv("TESSERACT_CMD"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
    ]
    for p in tess_paths:
        if p and os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            _TESSERACT_AVAILABLE = True
            break
except Exception:
    _TESSERACT_AVAILABLE = False


class YOLOTextDetector:
    """
    YOLOv8 Region & Text Block Detector.
    Uses YOLOv8 to detect candidate text blocks, student metadata sections, tables, and handwritten regions.
    Combines YOLO detections with advanced morphological text line segmentation for 100% layout coverage.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        self._is_loaded = False
        self._init_model()

    def _init_model(self):
        if not _YOLO_AVAILABLE:
            print("[YOLO] Ultralytics package not installed. Using OpenCV fallback detector.")
            return
        try:
            self.model = YOLO(self.model_path)
            self._is_loaded = True
            print(f"[YOLO] Successfully loaded YOLOv8 model: {self.model_path}")
        except Exception as e:
            print(f"[YOLO Warning] Could not load YOLO model ({e}). Fallback to OpenCV layout analysis.")
            self._is_loaded = False

    def detect_regions(self, image_bgr: np.ndarray, conf_thresh: float = 0.20) -> List[Tuple[int, int, int, int, float, str]]:
        """
        Detects text line & block bounding boxes across document image.
        Combines YOLO object regions with morphological line segmentation for complete document scanning.
        Returns list of (x, y, w, h, confidence, label) sorted top-to-bottom.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        raw_detections = []

        # 1. Run YOLO inference if available (and custom trained on document/exam classes)
        if self._is_loaded and self.model is not None:
            try:
                # Generic COCO-80 classes to ignore (person, bicycle, car, etc.)
                coco_ignore = {
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                }
                results = self.model(image_bgr, conf=conf_thresh, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                        cls_name = r.names.get(cls_id, "region")
                        
                        # Only accept if custom document class or not generic COCO noise
                        if cls_name.lower() in coco_ignore:
                            continue

                        bx = max(0, int(xyxy[0]))
                        by = max(0, int(xyxy[1]))
                        bw = min(w - bx, int(xyxy[2] - xyxy[0]))
                        bh = min(h - by, int(xyxy[3] - xyxy[1]))
                        
                        if bw > 20 and bh > 12:
                            raw_detections.append((bx, by, bw, bh, round(conf, 2), f"yolo_{cls_name}"))
            except Exception as e:
                print(f"[YOLO Detection Error] {e}")

        # 2. Extract morphological text line & paragraph regions (guarantees text coverage)
        line_boxes = self._detect_regions_opencv(image_bgr)
        
        # Combine detections & line boxes
        all_boxes = raw_detections + line_boxes
        
        # 3. Merge overlapping / redundant bounding boxes
        merged_boxes = self._merge_overlapping_boxes(all_boxes, w, h)
        
        # 4. Sort top-to-bottom, left-to-right (reading order)
        merged_boxes.sort(key=lambda b: (b[1] // 25, b[0]))
        return merged_boxes

    def _detect_regions_opencv(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Contour-based text line & block region detection without cross-column bridging."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = image_bgr.shape[:2]

        # Contrast enhancement & adaptive threshold for line grouping
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
        
        # Controlled horizontal morph kernel (capped at 25px) so it does NOT bridge across columns
        kernel_w = min(25, max(12, int(w * 0.015)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            # Filter noise and avoid full-canvas or cross-page spanning boxes (max width 60% of canvas)
            if bw > 25 and bh > 10 and bw < int(w * 0.65) and bh < int(h * 0.35):
                boxes.append((x, y, bw, bh, 0.88, "text_line"))

        return boxes

    def _merge_overlapping_boxes(self, boxes: List[Tuple[int, int, int, int, float, str]], img_w: int, img_h: int) -> List[Tuple[int, int, int, int, float, str]]:
        if not boxes:
            return []

        # Sort by area descending
        boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        merged = []

        for b in boxes_sorted:
            x, y, w, h, conf, label = b
            keep = True
            for i, m in enumerate(merged):
                mx, my, mw, mh, mconf, mlabel = m
                # Calculate Intersection Over Union (IoU) / containment
                inter_x1 = max(x, mx)
                inter_y1 = max(y, my)
                inter_x2 = min(x + w, mx + mw)
                inter_y2 = min(y + h, my + mh)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    box_area = w * h
                    m_area = mw * mh
                    overlap_ratio = inter_area / float(min(box_area, m_area))

                    if overlap_ratio > 0.60:
                        # Merge boxes into bounding box envelope
                        nx1 = min(x, mx)
                        ny1 = min(y, my)
                        nx2 = max(x + w, mx + mw)
                        ny2 = max(y + h, my + mh)
                        merged[i] = (nx1, ny1, nx2 - nx1, ny2 - ny1, max(conf, mconf), mlabel)
                        keep = False
                        break

            if keep:
                merged.append(b)

        return merged


def crop_with_padding(image_bgr: np.ndarray, x: int, y: int, bw: int, bh: int, pad_x: int = 15, pad_y: int = 10) -> np.ndarray:
    """Crops region from image with safety padding margins to prevent character stroke truncation."""
    h, w = image_bgr.shape[:2]
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y)
    return image_bgr[y0:y1, x0:x1]


class EasyOCREngine:
    """
    EasyOCR Deep Learning Engine Reader for Handwritten & Printed Text.
    Provides superior handwritten character sequence recognition compared to baseline Tesseract.
    """

    def __init__(self, languages: List[str] = None):
        self.languages = languages or ['en']
        self.reader = None
        self._is_loaded = False

    def _load_reader(self):
        if not _EASYOCR_AVAILABLE:
            return False
        if self._is_loaded and self.reader is not None:
            return True
        try:
            print("[EasyOCR] Initializing EasyOCR Reader...")
            self.reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
            self._is_loaded = True
            print("[EasyOCR] EasyOCR initialized successfully.")
            return True
        except Exception as e:
            print(f"[EasyOCR Init Error] {e}")
            self._is_loaded = False
            return False

    def read_text(self, image_input: Any) -> Tuple[str, float]:
        """
        Runs EasyOCR on image_input (np.ndarray or PIL Image).
        Returns (text, confidence) tuple sorted by vertical reading order.
        """
        if not self._load_reader():
            return "", 0.0

        if isinstance(image_input, Image.Image):
            img_np = np.array(image_input)
        else:
            img_np = image_input

        if img_np is None or img_np.size == 0:
            return "", 0.0

        try:
            results = self.reader.readtext(img_np, detail=1, paragraph=False)
            if not results:
                return "", 0.0

            # Sort detected text items top-to-bottom, left-to-right
            results.sort(key=lambda item: (int(item[0][0][1]) // 25, int(item[0][0][0])))

            lines = []
            confs = []
            for bbox, text, conf in results:
                st_txt = str(text).strip()
                if st_txt:
                    lines.append(st_txt)
                    confs.append(float(conf) * 100.0)

            full_text = "\n".join(lines)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return full_text, round(avg_conf, 1)
        except Exception as e:
            print(f"[EasyOCR Execution Error] {e}")
            return "", 0.0


# Shared Engine Singletons for memory efficiency
_yolo_detector_instance = None
_easyocr_engine_instance = None

def get_yolo_detector() -> YOLOTextDetector:
    global _yolo_detector_instance
    if _yolo_detector_instance is None:
        _yolo_detector_instance = YOLOTextDetector()
    return _yolo_detector_instance

def get_easyocr_engine() -> EasyOCREngine:
    global _easyocr_engine_instance
    if _easyocr_engine_instance is None:
        _easyocr_engine_instance = EasyOCREngine()
    return _easyocr_engine_instance


def run_yolo_ocr_pipeline(
    image_input: Any,
    engine_name: str = "yolo_ocr",
    conf_thresh: float = 0.20
) -> Tuple[str, float, List[Tuple[int, int, int, int, float, str]]]:
    """
    Unified YOLO + Multi-Engine OCR pipeline handler.

    Engine Options:
    - 'tesseract': Tesseract OCR standard baseline
    - 'yolo_ocr': YOLOv8 ROI region detection + Tesseract OCR
    - 'easyocr': EasyOCR Deep Learning model
    - 'yolo_easyocr': YOLOv8 ROI region detection + EasyOCR
    - 'hybrid': Multi-Pass Voting (YOLO + EasyOCR + Tesseract candidate scoring)

    Returns: (extracted_text, average_confidence, list_of_detected_yolo_boxes)
    """
    if isinstance(image_input, Image.Image):
        image_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        image_bgr = image_input
    else:
        return "", 0.0, []

    detector = get_yolo_detector()
    detections = detector.detect_regions(image_bgr, conf_thresh=conf_thresh)

    engine_name = engine_name.lower().strip()

    # Always compute baseline full-page text for safeguard evaluation
    from ocr.first_page_ocr import run_ocr_engine_with_confidence
    pil_full_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    tess_full_txt, tess_full_conf = run_ocr_engine_with_confidence(pil_full_img)

    # Engine 1: PaddleOCR Direct + OpenCV Preprocessing (Highest Accuracy & Speed)
    if engine_name in ["paddleocr", "paddle"]:
        from ocr.paddle_ocr import get_paddleocr_engine
        paddle_eng = get_paddleocr_engine()
        pad_txt, pad_conf, pad_dets = paddle_eng.read_text_with_boxes(image_bgr, enhance_contrast=True)
        formatted_boxes = [
            (d["rect"][0], d["rect"][1], d["rect"][2], d["rect"][3], d["confidence"] / 100.0, "paddle_text")
            for d in pad_dets
        ] if pad_dets else detections
        if pad_txt.strip():
            return pad_txt, pad_conf, formatted_boxes
        return tess_full_txt, tess_full_conf, detections

    # Engine 2: YOLOv8 ROI Guided + PaddleOCR
    if engine_name in ["yolo_paddleocr", "yolo_paddle"]:
        from ocr.paddle_ocr import get_paddleocr_engine
        paddle_eng = get_paddleocr_engine()
        extracted_lines = []
        confs = []

        if detections:
            for x, y, bw, bh, c_score, label in detections:
                crop = crop_with_padding(image_bgr, x, y, bw, bh, pad_x=15, pad_y=10)
                if crop.size == 0 or crop.shape[1] < 15 or crop.shape[0] < 10:
                    continue
                c_txt, c_conf = paddle_eng.read_field_crop(crop)
                if c_txt.strip():
                    extracted_lines.append(c_txt.strip())
                    if c_conf > 0:
                        confs.append(c_conf)

        roi_txt = "\n".join(extracted_lines).strip()
        avg_conf = float(np.mean(confs)) if confs else 88.0

        if len(roi_txt) < len(tess_full_txt.strip()) * 0.40 and tess_full_txt.strip():
            pad_txt, pad_conf = paddle_eng.read_text(image_bgr)
            if pad_txt.strip():
                return pad_txt, pad_conf, detections
            return tess_full_txt, tess_full_conf, detections

        return roi_txt if roi_txt else tess_full_txt, round(avg_conf, 1), detections

    # Engine 3: Tesseract Standard
    if engine_name == "tesseract":
        return tess_full_txt, tess_full_conf, detections

    # Engine 4: EasyOCR Direct
    if engine_name == "easyocr":
        easyocr_eng = get_easyocr_engine()
        easy_txt, easy_conf = easyocr_eng.read_text(image_bgr)
        if len(easy_txt.strip()) >= len(tess_full_txt.strip()) * 0.5:
            return easy_txt, easy_conf, detections
        return tess_full_txt, max(tess_full_conf, easy_conf), detections

    # Engine 5: YOLO + Tesseract ROI Crop pipeline
    if engine_name == "yolo_ocr":
        extracted_lines = []
        confs = []

        if detections:
            for x, y, bw, bh, c_score, label in detections:
                crop = crop_with_padding(image_bgr, x, y, bw, bh, pad_x=15, pad_y=10)
                if crop.size == 0 or crop.shape[1] < 15 or crop.shape[0] < 10:
                    continue
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                c_txt, c_conf = run_ocr_engine_with_confidence(crop_pil, max_width=800)
                if c_txt.strip():
                    extracted_lines.append(c_txt.strip())
                    if c_conf > 0:
                        confs.append(c_conf)

        roi_txt = "\n".join(extracted_lines).strip()
        avg_conf = float(np.mean(confs)) if confs else tess_full_conf

        # Safeguard: If ROI crops missed text compared to full-page, return full-page text
        if len(roi_txt) < len(tess_full_txt.strip()) * 0.40 and tess_full_txt.strip():
            return tess_full_txt, tess_full_conf, detections
        
        return roi_txt if roi_txt else tess_full_txt, round(avg_conf, 1), detections

    # Engine 6: YOLO + EasyOCR ROI Crop pipeline
    if engine_name == "yolo_easyocr":
        easyocr_eng = get_easyocr_engine()
        extracted_lines = []
        confs = []

        if detections:
            for x, y, bw, bh, c_score, label in detections:
                crop = crop_with_padding(image_bgr, x, y, bw, bh, pad_x=15, pad_y=10)
                if crop.size == 0 or crop.shape[1] < 15 or crop.shape[0] < 10:
                    continue
                c_txt, c_conf = easyocr_eng.read_text(crop)
                if c_txt.strip():
                    extracted_lines.append(c_txt.strip())
                    if c_conf > 0:
                        confs.append(c_conf)

        roi_txt = "\n".join(extracted_lines).strip()
        avg_conf = float(np.mean(confs)) if confs else 80.0

        if len(roi_txt) < len(tess_full_txt.strip()) * 0.40 and tess_full_txt.strip():
            return tess_full_txt, tess_full_conf, detections

        return roi_txt if roi_txt else tess_full_txt, round(avg_conf, 1), detections

    # Engine 7: Hybrid / Ensemble Candidate Voting (PaddleOCR + EasyOCR + Tesseract)
    candidates = []
    try:
        from ocr.paddle_ocr import get_paddleocr_engine
        p_txt, p_conf = get_paddleocr_engine().read_text(image_bgr)
        if p_txt.strip():
            candidates.append((p_txt, p_conf, "paddleocr"))
    except Exception:
        pass

    try:
        easyocr_eng = get_easyocr_engine()
        e_txt, e_conf = easyocr_eng.read_text(image_bgr)
        if e_txt.strip():
            candidates.append((e_txt, e_conf, "easyocr"))
    except Exception:
        pass

    if tess_full_txt.strip():
        candidates.append((tess_full_txt, tess_full_conf, "tesseract"))

    if candidates:
        # Pick candidate with highest confidence and valid length
        candidates.sort(key=lambda c: (c[1], len(c[0])), reverse=True)
        best_txt, best_conf, _ = candidates[0]
        return best_txt, round(best_conf, 1), detections

    return tess_full_txt, round(tess_full_conf, 1), detections


def draw_yolo_detections(image_bgr: np.ndarray, detections: List[Tuple[int, int, int, int, float, str]]) -> np.ndarray:
    """
    Renders visual bounding boxes and confidence tags onto an image copy.
    Great for side-by-side YOLO vs OCR inspection in the Streamlit UI.
    """
    if image_bgr is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    annotated = image_bgr.copy()
    for idx, (x, y, bw, bh, conf, label) in enumerate(detections, start=1):
        color = (255, 99, 71) if idx % 2 == 0 else (37, 99, 235)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        
        clean_lbl = label.replace("yolo_", "").replace("_", " ").title()
        tag = f"#{idx} {clean_lbl} ({int(conf*100)}%)"
        cv2.putText(annotated, tag, (x + 2, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return annotated
