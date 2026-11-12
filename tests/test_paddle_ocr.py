import unittest
import numpy as np
import cv2
from ocr.paddle_ocr import get_paddleocr_engine, draw_paddle_detections, PaddleOCREngine

class TestPaddleOCREngine(unittest.TestCase):
    def setUp(self):
        self.engine = get_paddleocr_engine()

    def test_singleton_engine(self):
        eng2 = get_paddleocr_engine()
        self.assertIs(self.engine, eng2)

    def test_opencv_preprocessing(self):
        # Create a test synthetic noisy image
        img = np.full((200, 300, 3), 240, dtype=np.uint8)
        processed = PaddleOCREngine.preprocess_with_opencv(img, enhance_contrast=True)
        self.assertEqual(processed.shape[:2], (200, 300))
        self.assertEqual(processed.dtype, np.uint8)

    def test_text_recognition_on_synthetic_opencv_image(self):
        # Create clean white canvas with crisp text using OpenCV
        img = np.full((120, 500, 3), 255, dtype=np.uint8)
        cv2.putText(img, "TEST QUESTION Q1 ANSWER", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        text, conf, detections = self.engine.read_text_with_boxes(img, enhance_contrast=True)
        self.assertTrue(len(text) > 0)
        self.assertTrue(conf > 50.0)
        self.assertTrue(any("TEST" in d["text"] or "QUESTION" in d["text"] or "ANSWER" in d["text"] for d in detections))

        # Test visualization overlay
        annotated = draw_paddle_detections(img, detections)
        self.assertEqual(annotated.shape, img.shape)

    def test_field_crop_reading(self):
        crop = np.full((50, 150, 3), 255, dtype=np.uint8)
        cv2.putText(crop, "CS501", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        txt, conf = self.engine.read_field_crop(crop)
        self.assertIn("CS501", txt.upper())

if __name__ == "__main__":
    unittest.main()
