import unittest
import numpy as np
from ocr.marks import parse_mark_value, detect_fraction_in_crop, extract_marks_table

class TestMarksFix(unittest.TestCase):
    def test_parse_mark_value_fractions(self):
        self.assertEqual(parse_mark_value("½", is_subpart=True), 0.5)
        self.assertEqual(parse_mark_value("1/2", is_subpart=True), 0.5)
        self.assertEqual(parse_mark_value("1½", is_subpart=True), 1.5)
        self.assertEqual(parse_mark_value("2½", is_subpart=True), 2.5)
        self.assertEqual(parse_mark_value("5½", is_subpart=False), 5.5)
        self.assertEqual(parse_mark_value("(5½)", is_subpart=False), 5.5)
        self.assertEqual(parse_mark_value("¼", is_subpart=True), 0.25)
        self.assertEqual(parse_mark_value("¾", is_subpart=True), 0.75)

    def test_parse_mark_value_valid_integers_and_decimals(self):
        # Ensure examiner marks are preserved without silent corruption
        self.assertEqual(parse_mark_value("1", is_subpart=True), 1.0)
        self.assertEqual(parse_mark_value("2", is_subpart=True), 2.0)
        self.assertEqual(parse_mark_value("3", is_subpart=True), 3.0)
        self.assertEqual(parse_mark_value("4", is_subpart=True), 4.0)
        self.assertEqual(parse_mark_value("3.5", is_subpart=True), 3.5)
        self.assertEqual(parse_mark_value("0.5", is_subpart=True), 0.5)
        self.assertEqual(parse_mark_value("0", is_subpart=True), 0.0)

    def test_fraction_contour_detection(self):
        # When fraction contour is detected on ambiguous multi-stroke text
        self.assertEqual(parse_mark_value("12", is_subpart=True, is_fraction_contour=True), 0.5)
        self.assertEqual(parse_mark_value("", is_subpart=True, is_fraction_contour=True), 0.5)

    def test_extract_marks_table_execution(self):
        # Blank canvas 1600x2200
        canvas = np.full((2200, 1600, 3), 255, dtype=np.uint8)
        question_results, total_marks, max_marks, overlay = extract_marks_table(canvas)
        self.assertIsInstance(question_results, list)
        self.assertEqual(len(question_results), 10)
        self.assertGreaterEqual(max_marks, 0.0)
        self.assertGreaterEqual(total_marks, 0.0)

if __name__ == "__main__":
    unittest.main()
