"""
Comprehensive OCR Accuracy Benchmark and Verification Test Suite
==================================================================
Measures field-by-field accuracy against verified ground-truth labels for real
examination booklet images. Validates:
1. Overlapping box rejection and neighbor bleed prevention
2. Handwritten student names & enrollment numbers
3. Date, section, class, and course title extraction
4. Decimal marks (3.5, 0.5), fractions (5 1/2), and blanks in marks table
5. Denominator preservation (18/30)
6. Rotation robustness (90°, 180°, 270°)
"""

import os
import json
import unittest
import numpy as np
import cv2
from PIL import Image
import difflib

from ocr.preprocessing import (
    convert_to_opencv,
    find_best_orthogonal_orientation,
    correct_document_perspective,
    remove_shadows_and_enhance,
    is_valid_crop
)
from ocr.first_page_ocr import process_first_page
from ocr.student_details import extract_student_details_page1
from ocr.marks import extract_marks_table, parse_mark_value, is_cell_blank, parse_grand_total_string
from template_config import NORMALIZED_WIDTH, NORMALIZED_HEIGHT

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth_labels.json")

def similarity(a: str, b: str) -> float:
    """Computes character sequence similarity ratio between 0.0 and 1.0."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, str(a).strip().lower(), str(b).strip().lower()).ratio()

class TestOCRAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            cls.benchmark_data = json.load(f)

    def test_decimal_and_fraction_mark_parsing(self):
        """Validates that marks decimals (3.5, 0.5, 6.5), fractions (5 1/2), and zeros are preserved."""
        cases = [
            ("3.5", 3.5),
            ("0.5", 0.5),
            ("6.5", 6.5),
            ("0", 0.0),
            ("1", 1.0),
            ("2", 2.0),
            ("5 1/2", 5.5),
            ("5½", 5.5),
            ("1/2", 0.5),
            ("½", 0.5),
            ("¼", 0.25),
            ("¾", 0.75),
            ("18/30", 18.0)
        ]
        for raw, expected in cases:
            parsed = parse_mark_value(raw, is_subpart=False)
            self.assertEqual(parsed, expected, f"Failed parsing mark '{raw}': got {parsed}, expected {expected}")

    def test_blank_cell_rejection(self):
        """Ensures truly blank cells are not hallucinated into non-zero marks."""
        # Pure white cell
        white_cell = np.full((60, 80, 3), 255, dtype=np.uint8)
        self.assertTrue(is_cell_blank(white_cell))

        # Blank cell with slight paper noise
        noisy_cell = np.full((60, 80, 3), 245, dtype=np.uint8)
        noise = np.random.randint(0, 10, (60, 80, 3), dtype=np.uint8)
        noisy_cell = cv2.subtract(noisy_cell, noise)
        self.assertTrue(is_cell_blank(noisy_cell))

        # Non-blank cell with actual handwritten mark
        ink_cell = np.full((60, 80, 3), 255, dtype=np.uint8)
        cv2.putText(ink_cell, "3.5", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 200), 2)
        self.assertFalse(is_cell_blank(ink_cell))

    def test_grand_total_with_denominator(self):
        """Verifies examiner written grand total extraction with denominator e.g. '18/30'."""
        num, denom, s = parse_grand_total_string("18/30")
        self.assertEqual(num, 18.0)
        self.assertEqual(denom, 30.0)
        self.assertEqual(s, "18/30")

        num, denom, s = parse_grand_total_string("5 1/2")
        self.assertEqual(num, 5.5)
        self.assertIsNone(denom)

        num, denom, s = parse_grand_total_string("6.5")
        self.assertEqual(num, 6.5)

    def test_rotation_handling(self):
        """Verifies orientation detection correctly handles 90, 180, and 270 degree rotations."""
        sample_file = os.path.join(ROOT_DIR, self.benchmark_data["samples"][0]["filename"])
        if not os.path.exists(sample_file):
            self.skipTest(f"Sample file not found: {sample_file}")

        img_bgr = cv2.imread(sample_file)
        
        # Test upright
        upright, ang0 = find_best_orthogonal_orientation(img_bgr, is_first_page=True)
        self.assertEqual(ang0, 0, "Upright image should be detected as 0°")
        
        # Test 180 degree upside-down rotation
        rot180 = cv2.rotate(img_bgr, cv2.ROTATE_180)
        upright180, ang180 = find_best_orthogonal_orientation(rot180, is_first_page=True)
        self.assertEqual(ang180, 180, "Upside-down image should be detected as 180°")
        diff180 = np.max(np.abs(upright180.astype(int) - img_bgr.astype(int)))
        self.assertEqual(diff180, 0, "Upright-corrected 180° image should match original upright image")

        # Test 90 degree clockwise rotation (landscape)
        rot90 = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        upright90, ang90 = find_best_orthogonal_orientation(rot90, is_first_page=True)
        self.assertEqual(ang90, 270, "90° CW image should require 270° rotation to become upright")
        self.assertEqual(upright90.shape, img_bgr.shape, "Corrected 90° CW image must have portrait dimensions")

        # Test 270 degree clockwise rotation (landscape)
        rot270 = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        upright270, ang270 = find_best_orthogonal_orientation(rot270, is_first_page=True)
        self.assertEqual(ang270, 90, "270° CW image should require 90° rotation to become upright")
        self.assertEqual(upright270.shape, img_bgr.shape, "Corrected 270° CW image must have portrait dimensions")

    def test_sample_1_flatbed_scan_accuracy(self):
        """Verifies field-by-field accuracy on flatbed scan (Sample 1: Bhavana Vundru)."""
        sample_info = self.benchmark_data["samples"][0]
        img_path = os.path.join(ROOT_DIR, sample_info["filename"])
        if not os.path.exists(img_path):
            self.skipTest(f"File missing: {img_path}")

        pil_img = Image.open(img_path)
        student_details, raw_text, conf = process_first_page(pil_img, detected_angle=0)
        gt = sample_info["ground_truth"]

        # Student Name
        self.assertTrue(similarity(student_details.student_name, gt["student_name"]) > 0.70,
                        f"Student name misread: {student_details.student_name} vs {gt['student_name']}")

        # Enrollment Number (must not leak printed header 'MINATION' and must match handwritten box)
        self.assertNotIn("MINATION", student_details.roll_number.upper(),
                         f"Enrollment leaked form label MINATION: {student_details.roll_number}")
        self.assertTrue(similarity(student_details.roll_number, gt["enrollment_no"]) > 0.80,
                        f"Enrollment mismatch: {student_details.roll_number} vs {gt['enrollment_no']}")

        # Section
        self.assertEqual(student_details.section.strip().upper(), gt["section"].upper())

        # Test Number
        self.assertIn("1", student_details.test_number)

        # Date
        self.assertIn("27/11/2025", student_details.exam_date)

        # Course Title & Code
        self.assertTrue("MACHINE" in student_details.course_title.upper() or "LEARNING" in student_details.course_title.upper())
        self.assertIn("DS804", student_details.subject_code.upper())

    def test_sample_2_phone_capture_accuracy(self):
        """Verifies perspective correction and OCR on phone camera capture (Sample 2: V. PARTHU)."""
        sample_info = self.benchmark_data["samples"][1]
        img_path = os.path.join(ROOT_DIR, sample_info["filename"])
        if not os.path.exists(img_path):
            self.skipTest(f"File missing: {img_path}")

        pil_img = Image.open(img_path)
        student_details, raw_text, conf = process_first_page(pil_img, detected_angle=0)
        gt = sample_info["ground_truth"]

        # Enrollment Number (must not contain MINATION, must reflect handwritten box)
        self.assertNotIn("MINATION", student_details.roll_number.upper())
        self.assertTrue(similarity(student_details.roll_number, gt["enrollment_no"]) > 0.50,
                        f"Enrollment mismatch on phone capture: {student_details.roll_number} vs {gt['enrollment_no']}")

        # Section
        self.assertEqual(student_details.section.strip().upper(), "H")

        # Test Number
        self.assertIn("2", student_details.test_number)

        # Course Code
        self.assertIn("CS203", student_details.subject_code.upper())

        # Course Title (fuzzy matched to syllabus)
        self.assertTrue(similarity(student_details.course_title, gt["course_title"]) > 0.60)

    def test_sample_4_denominator_grand_total(self):
        """Verifies black-and-white booklet capture with 18/30 denominator."""
        sample_info = self.benchmark_data["samples"][3]
        img_path = os.path.join(ROOT_DIR, sample_info["filename"])
        if not os.path.exists(img_path):
            self.skipTest(f"File missing: {img_path}")

        pil_img = Image.open(img_path)
        student_details, raw_text, conf = process_first_page(pil_img, detected_angle=0)

        # Enrollment Number (must not leak MINATION, must correctly localize handwritten box)
        self.assertNotIn("MINATION", student_details.roll_number.upper())
        self.assertTrue(similarity(student_details.roll_number, "24STUCHH010010") > 0.55,
                        f"Enrollment mismatch on sample 4: {student_details.roll_number}")
        self.assertTrue("24STUCHH" in student_details.roll_number.upper(),
                        f"Expected batch prefix 24STUCHH in: {student_details.roll_number}")

        # Section
        self.assertEqual(student_details.section.strip().upper(), "D")

        # Grand Total (must NOT confuse with Examiner Signature and must capture '18')
        gt_val = getattr(student_details, "grand_total", "")
        self.assertNotIn("SIGNATURE", gt_val.upper(), "Grand total confused with Examiner Signature!")
        self.assertNotIn("EXAMINER", gt_val.upper(), "Grand total confused with Examiner Signature!")
        self.assertTrue("18" in gt_val, f"Grand total not extracted properly: {gt_val}")

if __name__ == "__main__":
    unittest.main()
