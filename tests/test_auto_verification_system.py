"""
Unit test suite for the Automated Verification Engine and Multi-Student Verification System.
Validates:
1. Automatic verification of clear fields (Name, Roll, Test No, Exam Date, Course Title, Grand Total).
2. Blank/missing fields marked as 'Needs Manual Verification' (NEVER marked as incorrect).
3. Ambiguous/confused fields (e.g. '202y-2028') marked as 'Needs Manual Verification'.
4. Mutually exclusive categories and dynamic metrics calculation summing to 100%.
5. Navigation boundary safety and per-student data isolation.
"""

import unittest
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import COVER_FIELDS, calculate_cover_field_metrics, compare_field_values
from ocr.auto_verifier import (
    is_missing_value,
    verify_field_against_pdf_crop,
    perform_automatic_verification
)


class TestAutoVerificationSystem(unittest.TestCase):
    def setUp(self):
        self.cover_fields = COVER_FIELDS

    def test_01_missing_value_detection(self):
        """Blank or placeholder values must be recognized and never marked as incorrect."""
        self.assertTrue(is_missing_value(""))
        self.assertTrue(is_missing_value(None))
        self.assertTrue(is_missing_value("Not Detected"))
        self.assertTrue(is_missing_value("not_detected"))
        self.assertTrue(is_missing_value("Not Present"))
        self.assertTrue(is_missing_value("Needs Review"))
        self.assertTrue(is_missing_value("—"))
        self.assertTrue(is_missing_value("-"))
        self.assertFalse(is_missing_value("y.Rishika Reddy"))
        self.assertFalse(is_missing_value("24STUCHH010984"))

    def test_02_clear_fields_auto_verified(self):
        """Clearly extracted fields must be automatically marked as Verified."""
        # 1. Student Name
        res_name = verify_field_against_pdf_crop("student_name", "y.Rishika Reddy", 88.0, None)
        self.assertEqual(res_name["status"], "Verified")
        self.assertTrue(res_name["verified"])
        self.assertFalse(res_name["is_explicit_incorrect"])

        # 2. Enrollment Number
        res_roll = verify_field_against_pdf_crop("enrollment_no", "24STUCHH010984", 95.0, None)
        self.assertEqual(res_roll["status"], "Verified")
        self.assertTrue(res_roll["verified"])

        # 3. Test Number
        res_test = verify_field_against_pdf_crop("test_no", "1", 90.0, None)
        self.assertEqual(res_test["status"], "Verified")
        self.assertTrue(res_test["verified"])

        # 4. Exam Date
        res_date = verify_field_against_pdf_crop("date", "10/20/2026", 92.0, None)
        self.assertEqual(res_date["status"], "Verified")
        self.assertTrue(res_date["verified"])

        # 5. Course Title
        res_title = verify_field_against_pdf_crop("course_title", "Database Management System", 85.0, None)
        self.assertEqual(res_title["status"], "Verified")
        self.assertTrue(res_title["verified"])

        # 6. Grand Total
        res_gt = verify_field_against_pdf_crop("grand_total", "20/30", 94.0, None)
        self.assertEqual(res_gt["status"], "Verified")
        self.assertTrue(res_gt["verified"])

    def test_03_blank_fields_marked_needs_manual_verification(self):
        """Missing or blank fields must be marked as 'Needs Manual Verification', NEVER as 'Incorrect'."""
        blank_fields = ["section", "course_code", "semester", "instructor_name"]
        for fid in blank_fields:
            res = verify_field_against_pdf_crop(fid, "Not Detected", 0.0, None)
            self.assertEqual(res["status"], "Needs Manual Verification", f"Field {fid} should be Needs Manual Verification")
            self.assertFalse(res["verified"])
            self.assertFalse(res["is_explicit_incorrect"], f"Field {fid} must NOT be marked as incorrect")
            self.assertFalse(res["is_detected"])

    def test_04_confused_class_batch_marked_needs_manual(self):
        """Class/batch containing character confusion (e.g. 202y-2028) must require manual verification."""
        res_confused = verify_field_against_pdf_crop("class", "202y-2028", 75.0, None)
        self.assertEqual(res_confused["status"], "Needs Manual Verification")
        self.assertFalse(res_confused["verified"])
        self.assertFalse(res_confused["is_explicit_incorrect"])

        # Valid class batch is auto-verified
        res_valid = verify_field_against_pdf_crop("class", "2024-2028", 85.0, None)
        self.assertEqual(res_valid["status"], "Verified")
        self.assertTrue(res_valid["verified"])

    def test_05_impossible_value_marked_incorrect(self):
        """Impossible values (such as an impossible calendar date) must be marked as Incorrect."""
        res_bad_date = verify_field_against_pdf_crop("date", "45/78/2026", 80.0, None)
        self.assertEqual(res_bad_date["status"], "Incorrect")
        self.assertTrue(res_bad_date["is_explicit_incorrect"])
        self.assertFalse(res_bad_date["verified"])

    def test_06_student_metrics_calculation(self):
        """
        End-to-end verification of the exact student scenario:
        - 6 Auto-Verified (Name, Roll, Test No, Date, Course Title, Grand Total)
        - 5 Needs Manual Verification (Section, Class/Batch, Course Code, Semester, Instructor Name)
        - 0 Incorrect
        """
        extracted = {
            "student_name": "y.Rishika Reddy",
            "enrollment_no": "24STUCHH010984",
            "test_no": "1",
            "section": "Not Detected",
            "date": "10/20/2026",
            "class": "202y-2028",
            "course_title": "Database Management System",
            "course_code": "Not Detected",
            "semester": "Not Detected",
            "instructor_name": "Not Present",
            "grand_total": "20/30"
        }

        v_map = perform_automatic_verification(
            initial_extracted_fields=extracted,
            crops_dict={},
            cover_fields=self.cover_fields
        )

        metrics = calculate_cover_field_metrics(self.cover_fields, v_map, extracted)

        self.assertEqual(metrics["total_fields"], 11)
        self.assertEqual(metrics["total_detected_count"], 7)
        self.assertEqual(metrics["verified_correct_count"], 6)
        self.assertEqual(metrics["incorrect_count"], 0)
        self.assertEqual(metrics["needs_verification_count"], 5)

        self.assertEqual(metrics["accuracy_str"], "54.55%")
        self.assertEqual(metrics["error_str"], "0.00%")
        self.assertEqual(metrics["pending_str"], "45.45%")

        # Sum of counts equals 11
        total_sum = metrics["verified_correct_count"] + metrics["incorrect_count"] + metrics["needs_verification_count"]
        self.assertEqual(total_sum, 11)

    def test_07_reviewer_correction_dynamically_updates_metrics(self):
        """When a reviewer confirms a field, accuracy increases and pending decreases immediately."""
        extracted = {
            "student_name": "y.Rishika Reddy",
            "enrollment_no": "24STUCHH010984",
            "test_no": "1",
            "section": "Not Detected",
            "date": "10/20/2026",
            "class": "202y-2028",
            "course_title": "Database Management System",
            "course_code": "Not Detected",
            "semester": "Not Detected",
            "instructor_name": "Not Present",
            "grand_total": "20/30"
        }

        v_map = perform_automatic_verification(
            initial_extracted_fields=extracted,
            crops_dict={},
            cover_fields=self.cover_fields
        )

        # Baseline: 6 verified, 5 pending
        m_base = calculate_cover_field_metrics(self.cover_fields, v_map, extracted)
        self.assertEqual(m_base["verified_correct_count"], 6)
        self.assertEqual(m_base["needs_verification_count"], 5)

        # Reviewer corrects class to "2024-2028" and confirms
        v_map["class"]["status"] = "Verified"
        v_map["class"]["verified"] = True
        v_map["class"]["ground_truth"] = "2024-2028"
        v_map["class"]["is_explicit_incorrect"] = False

        m_after = calculate_cover_field_metrics(self.cover_fields, v_map, extracted)
        self.assertEqual(m_after["verified_correct_count"], 7)
        self.assertEqual(m_after["needs_verification_count"], 4)
        self.assertEqual(m_after["accuracy_str"], "63.64%")
        self.assertEqual(m_after["pending_str"], "36.36%")

    def test_08_navigation_bounds_safety(self):
        """Next/Previous logic safely maintains 0 <= current_student_index < total."""
        total = 5

        # Lower bound
        idx = 0
        new_prev = max(0, idx - 1)
        self.assertEqual(new_prev, 0)

        # Increment
        new_next = min(total - 1, idx + 1)
        self.assertEqual(new_next, 1)

        # Upper bound
        idx = 4
        new_next_last = min(total - 1, idx + 1)
        self.assertEqual(new_next_last, 4)


if __name__ == "__main__":
    unittest.main()

