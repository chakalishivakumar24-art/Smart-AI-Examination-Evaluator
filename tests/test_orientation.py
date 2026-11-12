"""
Unit and Integration Tests for Image Orientation Detection, Auto-Rotation, and Coordinate Transformations.
Validates:
1. Upright image preservation (0° detected, not blindly rotated)
2. 180° upside-down image detection and upright correction
3. 90° sideways image detection and upright correction
4. 270° sideways image detection and upright correction
5. Forward and inverse coordinate bounding box transformations
6. OCR accuracy parity across 0°, 90°, 180°, and 270° rotations
"""

import os
import cv2
import json
import unittest
import numpy as np
from PIL import Image

from ocr.orientation import (
    find_best_orthogonal_orientation,
    detect_orientation_angle,
    detect_and_rotate_image,
    rotate_image_angle,
    transform_bbox,
    inverse_transform_bbox
)
from ocr.first_page_ocr import process_first_page
from ocr.preprocessing import convert_to_opencv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth_labels.json")


class TestImageOrientation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            cls.benchmark_data = json.load(f)
        cls.sample_path = os.path.join(ROOT_DIR, cls.benchmark_data["samples"][0]["filename"])
        cls.orig_bgr = cv2.imread(cls.sample_path)
        assert cls.orig_bgr is not None, f"Could not read sample image: {cls.sample_path}"

    def test_bbox_transformations(self):
        """Validates that transform_bbox and inverse_transform_bbox are exact mathematical inverses."""
        orig_w, orig_h = 1600, 2200
        test_boxes = [
            (100, 200, 300, 400),
            (0, 0, 100, 100),
            (1100, 60, 430, 115),
            (185, 525, 435, 85)
        ]
        for angle in [0, 90, 180, 270]:
            for box in test_boxes:
                transformed = transform_bbox(box, orig_w, orig_h, angle)
                recovered = inverse_transform_bbox(transformed, orig_w, orig_h, angle)
                self.assertEqual(box, recovered, f"Failed round-trip at {angle}°: {box} != {recovered}")

    def test_upright_image_preservation(self):
        """Verifies that an already upright image is detected as 0° and preserved without rotation."""
        upright_img, angle = find_best_orthogonal_orientation(self.orig_bgr, is_first_page=True)
        self.assertEqual(angle, 0, f"Expected 0° for upright image, got {angle}°")
        self.assertEqual(upright_img.shape, self.orig_bgr.shape)

    def test_180_degree_rotation_correction(self):
        """Verifies that an upside-down (180°) image is detected as 180° and rotated back to upright."""
        rot180 = cv2.rotate(self.orig_bgr, cv2.ROTATE_180)
        upright_img, angle = find_best_orthogonal_orientation(rot180, is_first_page=True)
        self.assertEqual(angle, 180, f"Expected 180° detection, got {angle}°")
        # Rotated 180 twice yields original image
        diff = np.max(np.abs(upright_img.astype(int) - self.orig_bgr.astype(int)))
        self.assertEqual(diff, 0, "Corrected 180° image differs from original upright image")

    def test_90_degree_clockwise_rotation_correction(self):
        """Verifies that a 90° CW sideways image is detected and corrected to upright portrait."""
        rot90 = cv2.rotate(self.orig_bgr, cv2.ROTATE_90_CLOCKWISE)
        upright_img, angle = find_best_orthogonal_orientation(rot90, is_first_page=True)
        # rot90 needs 270° CW rotation to become upright
        self.assertEqual(angle, 270, f"Expected 270° angle to fix 90° CW image, got {angle}°")
        self.assertEqual(upright_img.shape, self.orig_bgr.shape)

    def test_270_degree_clockwise_rotation_correction(self):
        """Verifies that a 270° CW sideways image is detected and corrected to upright portrait."""
        rot270 = cv2.rotate(self.orig_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        upright_img, angle = find_best_orthogonal_orientation(rot270, is_first_page=True)
        # rot270 needs 90° CW rotation to become upright
        self.assertEqual(angle, 90, f"Expected 90° angle to fix 270° CW image, got {angle}°")
        self.assertEqual(upright_img.shape, self.orig_bgr.shape)

    def test_ocr_parity_across_all_rotations(self):
        """Verifies that student details extraction yields identical results on 0°, 90°, 180°, and 270° rotations."""
        angles_to_test = [
            (0, self.orig_bgr),
            (180, cv2.rotate(self.orig_bgr, cv2.ROTATE_180)),
            (90, cv2.rotate(self.orig_bgr, cv2.ROTATE_90_CLOCKWISE)),
            (270, cv2.rotate(self.orig_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE))
        ]

        gt = self.benchmark_data["samples"][0]["ground_truth"]

        for input_rot, img_test in angles_to_test:
            pil_input = Image.fromarray(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
            details, raw_text, conf = process_first_page(pil_input)
            
            # Student Name must match Bhavana Vundru
            self.assertTrue("BHAVANA" in details.student_name.upper(),
                            f"At input rotation {input_rot}°, name was '{details.student_name}'")
            # Enrollment must match 23STUCHH010066
            self.assertTrue("23STUCHH010066" in details.roll_number.upper(),
                            f"At input rotation {input_rot}°, roll was '{details.roll_number}'")
            # Section must match C
            self.assertEqual(details.section.strip().upper(), "C",
                             f"At input rotation {input_rot}°, section was '{details.section}'")


if __name__ == "__main__":
    unittest.main()
