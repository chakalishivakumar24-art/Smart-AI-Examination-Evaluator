"""
PaddleOCR + OpenCV High-Performance Document & Handwritten Text Recognition Module.

Provides:
- Native OpenCV NumPy array processing and image enhancement (CLAHE, Bilateral filtering, Unsharp masking).
- PaddleOCR Deep Learning recognition pipeline for examination booklets.
- Reading order sorting, bounding box extraction, and confidence scoring.
- Small ROI field crop recognition for metadata and marks grid cells.
- Visual detection overlay rendering with OpenCV.
"""

import os
import cv2
import time
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional

# Ensure torch is imported prior to PaddleOCR on Windows to prevent DLL collision
try:
    import torch
except ImportError:
    pass

_PADDLEOCR_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    _PADDLEOCR_AVAILABLE = True
except Exception:
    _PADDLEOCR_AVAILABLE = False


class PaddleOCREngine:
    """
    PaddleOCR Deep Learning Engine Reader for Document & Handwritten Text.
    Optimized for high-speed inference, multi-orientation detection, and seamless OpenCV integration.
    """

    def __init__(self, lang: str = "en", use_angle_cls: bool = False):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.ocr_reader = None
        self._is_loaded = False

    def _load_reader(self) -> bool:
        if not _PADDLEOCR_AVAILABLE:
            return False
        if self._is_loaded and self.ocr_reader is not None:
            return True
        try:
            print("[PaddleOCR] Initializing PaddleOCR Reader (PP-OCRv4)...")
            # Note: show_log=False suppresses verbose terminal spam in modern PaddleOCR
            self.ocr_reader = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=False,
                enable_mkldnn=False
            )
            self._is_loaded = True
            print("[PaddleOCR] PaddleOCR initialized successfully.")
            return True
        except Exception as e:
            print(f"[PaddleOCR Init Error] {e}")
            self._is_loaded = False
            return False

    @staticmethod
    def preprocess_with_opencv(
        image_input: Any,
        enhance_contrast: bool = True,
        max_dimension: int = 2000
    ) -> np.ndarray:
        """
        OpenCV Preprocessing Pipeline tailored for PaddleOCR:
        1. Format conversion to OpenCV BGR NumPy array.
        2. Proportional resizing to optimal canvas resolution (prevents OOM on 600+ DPI scans).
        3. LAB CLAHE contrast enhancement for faint pencil/ink strokes.
        4. Bilateral filtering to smooth paper background grain while preserving stroke edges.
        5. Unsharp masking to sharpen character contours.
        """
        # 1. Format conversion
        if isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            else:
                img_bgr = image_input.copy()
        elif isinstance(image_input, (str, bytes)):
            if isinstance(image_input, str):
                img_bgr = cv2.imread(image_input)
            else:
                nparr = np.frombuffer(image_input, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Could not decode image input")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        if img_bgr is None or img_bgr.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        h, w = img_bgr.shape[:2]

        # 2. Resize if oversized
        if max(h, w) > max_dimension:
            scale = max_dimension / float(max(h, w))
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        if not enhance_contrast:
            return img_bgr

        # 3. CLAHE Contrast enhancement in LAB color space
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_chan)
        lab_enhanced = cv2.merge((l_enhanced, a_chan, b_chan))
        enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 4. Bilateral filter to denoise paper background without blurring handwritten strokes
        denoised = cv2.bilateralFilter(enhanced_bgr, d=7, sigmaColor=50, sigmaSpace=50)

        # 5. Unsharp mask for crisper text edges
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        sharpened = cv2.addWeighted(denoised, 1.25, gaussian, -0.25, 0)

        return sharpened

    def read_text(self, image_input: Any, enhance_contrast: bool = True) -> Tuple[str, float]:
        """
        Runs PaddleOCR on image_input (PIL Image or NumPy array).
        Returns: (full_text, average_confidence)
        """
        full_text, avg_conf, _ = self.read_text_with_boxes(image_input, enhance_contrast=enhance_contrast)
        return full_text, avg_conf

    def read_text_with_boxes(
        self,
        image_input: Any,
        enhance_contrast: bool = True
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Runs PaddleOCR detection & recognition on image.
        Sorts detected lines top-to-bottom, left-to-right (natural reading order).
        Returns: (full_text, average_confidence, list_of_detection_dicts)
        """
        if not self._load_reader():
            return "", 0.0, []

        img_cv = self.preprocess_with_opencv(image_input, enhance_contrast=enhance_contrast)
        if img_cv is None or img_cv.size == 0:
            return "", 0.0, []

        try:
            results = self.ocr_reader.ocr(img_cv, cls=False)
            if not results or not results[0]:
                return "", 0.0, []

            detections = []
            page_results = results[0]

            for item in page_results:
                if not item or len(item) < 2:
                    continue
                bbox_pts = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = item[1]  # (text, confidence)
                
                text = str(text_info[0]).strip()
                if not text:
                    continue
                    
                conf = float(text_info[1]) * 100.0 if text_info[1] <= 1.0 else float(text_info[1])

                # Calculate bounding rect (x, y, w, h)
                xs = [pt[0] for pt in bbox_pts]
                ys = [pt[1] for pt in bbox_pts]
                rx = int(min(xs))
                ry = int(min(ys))
                rw = int(max(xs) - rx)
                rh = int(max(ys) - ry)

                detections.append({
                    "points": bbox_pts,
                    "rect": (rx, ry, rw, rh),
                    "text": text,
                    "confidence": round(conf, 1)
                })

            if not detections:
                return "", 0.0, []

            # Sort top-to-bottom, left-to-right (bucket lines within 25px vertical band)
            detections.sort(key=lambda d: (d["rect"][1] // 25, d["rect"][0]))

            lines = [d["text"] for d in detections]
            confs = [d["confidence"] for d in detections]

            full_text = "\n".join(lines).strip()
            avg_conf = float(np.mean(confs)) if confs else 0.0

            return full_text, round(avg_conf, 1), detections

        except Exception as e:
            print(f"[PaddleOCR Execution Error] {e}")
            return "", 0.0, []

    def read_field_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Tailored PaddleOCR extraction for small ROI fields (e.g. Student Name, Roll No, Marks Cell).
        Applies OpenCV white border padding and upscaling.
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
            return "", 0.0

        if not self._load_reader():
            return "", 0.0

        # Add 12px white padding with OpenCV to prevent character boundary clipping
        padded = cv2.copyMakeBorder(crop_bgr, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # 2x Cubic Upscaling for fine handwritten lines
        h, w = padded.shape[:2]
        if h < 80 or w < 120:
            padded = cv2.resize(padded, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        text, conf = self.read_text(padded, enhance_contrast=True)
        # Flatten multiline crops to single line for field values
        cleaned_text = " ".join([ln.strip() for ln in text.split("\n") if ln.strip()]).strip()
        return cleaned_text, conf


# Shared Singleton Instance
_paddle_engine_instance: Optional[PaddleOCREngine] = None


def get_paddleocr_engine() -> PaddleOCREngine:
    """Returns singleton PaddleOCREngine instance with lazy loading."""
    global _paddle_engine_instance
    if _paddle_engine_instance is None:
        _paddle_engine_instance = PaddleOCREngine()
    return _paddle_engine_instance


def draw_paddle_detections(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    Renders visual bounding polygons/boxes and confidence tags onto an OpenCV image copy.
    """
    if image_bgr is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    annotated = image_bgr.copy()
    for idx, d in enumerate(detections, start=1):
        rx, ry, rw, rh = d["rect"]
        conf = d["confidence"]
        text = d["text"]

        # Color alternation for visual clarity
        color = (0, 165, 255) if idx % 2 == 0 else (50, 205, 50)  # Orange / Green in BGR

        # Draw rectangle
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), color, 2)

        # Label tag
        tag = f"#{idx} {text} ({int(conf)}%)"
        tag_y = max(18, ry - 6)
        cv2.putText(annotated, tag, (rx + 2, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return annotated
