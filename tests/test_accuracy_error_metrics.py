"""
Unit Test Suite for Dynamic Cover-Page Field Accuracy, Error, and Pending Verification Metrics.
Validates:
1. Dynamic calculations across all 11 fields.
2. The three mutually exclusive statuses: 'Verified Correct', 'Incorrect', 'Needs Verification'.
3. Strict formulas:
   - Accuracy (%) = (Verified Correct / 11) * 100
   - Error (%) = (Confirmed Incorrect / 11) * 100
   - Pending Verification (%) = (Needs Verification / 11) * 100
4. Verification of the exact example:
   - 6 Verified Correct, 0 Incorrect, 5 Needs Verification -> 54.55%, 0.00%, 45.45%.
5. Unverified fields are never counted as correct or incorrect.
6. Missing/undetected OCR values are never counted as verified correct.
7. Sum of status counts is always exactly 11.
8. Safe handling of 0 fields without division-by-zero errors.
"""

import unittest
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import COVER_FIELDS, compare_field_values, calculate_cover_field_metrics


class TestAccuracyErrorMetrics(unittest.TestCase):
    def setUp(self):
        self.cover_fields = COVER_FIELDS

    def test_total_cover_fields_is_eleven(self):
        """Requirement 1 & 7: Exactly 11 cover page fields must be defined."""
        self.assertEqual(len(self.cover_fields), 11)
        expected_ids = [
            "student_name", "enrollment_no", "test_no", "section", "date",
            "class", "course_title", "course_code", "semester", "instructor_name", "grand_total"
        ]
        actual_ids = [f["id"] for f in self.cover_fields]
        self.assertEqual(actual_ids, expected_ids)

    def test_user_example_six_correct_five_pending(self):
        """
        Requirement 4:
        Total fields: 11
        Verified Correct: 6
        Incorrect: 0
        Needs Verification: 5
        Expected: Accuracy = 54.55%, Error = 0.00%, Pending Verification = 45.45%
        """
        initial_extracted = {
            "student_name": "Bhavana Vundru",
            "enrollment_no": "23STUCHH010066",
            "section": "C",
            "class": "2023 - 27",
            "course_title": "Machine Learning",
            "date": "27/11/2025",
            "test_no": "1",
            "course_code": "DS804",
            "semester": "V",
            "instructor_name": "",
            "grand_total": ""
        }

        # 6 fields verified and matching ground truth; 5 fields unverified (Needs Verification)
        v_map = {
            # 6 Verified Correct
            "student_name": {"verified": True, "ground_truth": "Bhavana Vundru"},
            "enrollment_no": {"verified": True, "ground_truth": "23STUCHH010066"},
            "section": {"verified": True, "ground_truth": "C"},
            "class": {"verified": True, "ground_truth": "2023 - 27"},
            "course_title": {"verified": True, "ground_truth": "Machine Learning"},
            "date": {"verified": True, "ground_truth": "27/11/2025"},
            # 5 Needs Verification
            "test_no": {"verified": False, "ground_truth": ""},
            "course_code": {"verified": False, "ground_truth": ""},
            "semester": {"verified": False, "ground_truth": ""},
            "instructor_name": {"verified": False, "ground_truth": ""},
            "grand_total": {"verified": False, "ground_truth": ""}
        }

        metrics = calculate_cover_field_metrics(self.cover_fields, v_map, initial_extracted)

        # Count assertions
        self.assertEqual(metrics["total_fields"], 11)
        self.assertEqual(metrics["verified_correct_count"], 6)
        self.assertEqual(metrics["incorrect_count"], 0)
        self.assertEqual(metrics["needs_verification_count"], 5)
        self.assertEqual(
            metrics["verified_correct_count"] + metrics["incorrect_count"] + metrics["needs_verification_count"],
            11
        )

        # Percentage assertions
        self.assertEqual(metrics["accuracy_pct"], 54.55)
        self.assertEqual(metrics["error_pct"], 0.00)
        self.assertEqual(metrics["pending_pct"], 45.45)

        # Formatted string assertions
        self.assertEqual(metrics["accuracy_str"], "54.55%")
        self.assertEqual(metrics["error_str"], "0.00%")
        self.assertEqual(metrics["pending_str"], "45.45%")

        # Subtitle labels
        self.assertEqual(metrics["accuracy_sub"], "6 of 11 fields verified correct")
        self.assertEqual(metrics["error_sub"], "0 of 11 fields confirmed incorrect")
        self.assertEqual(metrics["pending_sub"], "5 of 11 fields need verification")

    def test_missing_or_undetected_cannot_be_verified_correct(self):
        """
        Requirement 6: Missing or undetected values must never be counted as verified correct,
        even if a human manually checked 'verified'.
        """
        initial_extracted = {
            "student_name": "Not Detected",
            "instructor_name": "",
            "grand_total": None
        }
        v_map = {
            # User checked verified and entered ground truth, but OCR missed it
            "student_name": {"verified": True, "ground_truth": "Bhavana Vundru"},
            "instructor_name": {"verified": True, "ground_truth": "Dr. Sharma"},
            "grand_total": {"verified": True, "ground_truth": "25/30"}
        }
        for f in self.cover_fields:
            if f["id"] not in v_map:
                v_map[f["id"]] = {"verified": False, "ground_truth": ""}

        metrics = calculate_cover_field_metrics(self.cover_fields, v_map, initial_extracted)

        # Missing values must be counted as Incorrect, not Verified Correct
        self.assertEqual(metrics["field_status_map"]["student_name"], "Incorrect")
        self.assertEqual(metrics["field_status_map"]["instructor_name"], "Incorrect")
        self.assertEqual(metrics["field_status_map"]["grand_total"], "Incorrect")
        self.assertEqual(metrics["verified_correct_count"], 0)
        self.assertEqual(metrics["incorrect_count"], 3)
        self.assertEqual(metrics["needs_verification_count"], 8)

    def test_unverified_fields_not_counted_as_correct_or_incorrect(self):
        """
        Requirement 5: Fields needing verification must not be marked as correct or incorrect.
        """
        initial_extracted = {f["id"]: "Some Value" for f in self.cover_fields}
        # All 11 unverified
        v_map = {f["id"]: {"verified": False, "ground_truth": "Some Value"} for f in self.cover_fields}

        metrics = calculate_cover_field_metrics(self.cover_fields, v_map, initial_extracted)

        self.assertEqual(metrics["verified_correct_count"], 0)
        self.assertEqual(metrics["incorrect_count"], 0)
        self.assertEqual(metrics["needs_verification_count"], 11)
        self.assertEqual(metrics["accuracy_pct"], 0.00)
        self.assertEqual(metrics["error_pct"], 0.00)
        self.assertEqual(metrics["pending_pct"], 100.00)

    def test_dynamic_updates_when_field_status_changes(self):
        """
        Requirement 8: Ensure stats update dynamically when user corrects or verifies a field.
        """
        initial_extracted = {f["id"]: f"Val_{f['id']}" for f in self.cover_fields}
        v_map = {f["id"]: {"verified": True, "ground_truth": f"Val_{f['id']}"} for f in self.cover_fields}

        # Step 1: All 11 verified correct
        m1 = calculate_cover_field_metrics(self.cover_fields, v_map, initial_extracted)
        self.assertEqual(m1["verified_correct_count"], 11)
        self.assertEqual(m1["incorrect_count"], 0)
        self.assertEqual(m1["needs_verification_count"], 0)
        self.assertEqual(m1["accuracy_pct"], 100.00)
        self.assertEqual(m1["error_pct"], 0.00)
        self.assertEqual(m1["pending_pct"], 0.00)

        # Step 2: User corrects 2 fields with different ground truth (2 incorrect)
        v_map["student_name"]["ground_truth"] = "Corrected Name"
        v_map["date"]["ground_truth"] = "Corrected Date"
        m2 = calculate_cover_field_metrics(self.cover_fields, v_map, initial_extracted)
        self.assertEqual(m2["verified_correct_count"], 9)
        self.assertEqual(m2["incorrect_count"], 2)
        self.assertEqual(m2["needs_verification_count"], 0)
        self.assertEqual(m2["accuracy_pct"], 81.82)
        self.assertEqual(m2["error_pct"], 18.18)
        self.assertEqual(m2["pending_pct"], 0.00)

        # Step 3: User unchecks 3 fields (they become Needs Verification)
        v_map["test_no"]["verified"] = False
        v_map["section"]["verified"] = False
        v_map["class"]["verified"] = False
        m3 = calculate_cover_field_metrics(self.cover_fields, v_map, initial_extracted)
        self.assertEqual(m3["verified_correct_count"], 6)
        self.assertEqual(m3["incorrect_count"], 2)
        self.assertEqual(m3["needs_verification_count"], 3)
        self.assertEqual(m3["accuracy_pct"], 54.55)
        self.assertEqual(m3["error_pct"], 18.18)
        self.assertEqual(m3["pending_pct"], 27.27)

    def test_safe_zero_fields_division_by_zero(self):
        """
        Requirement 11: Handle zero fields safely without division-by-zero errors.
        """
        metrics = calculate_cover_field_metrics([], {}, {})
        self.assertEqual(metrics["total_fields"], 0)
        self.assertEqual(metrics["accuracy_pct"], 0.0)
        self.assertEqual(metrics["error_pct"], 0.0)
        self.assertEqual(metrics["pending_pct"], 0.0)
        self.assertEqual(metrics["accuracy_str"], "0.00%")
        self.assertEqual(metrics["error_str"], "0.00%")
        self.assertEqual(metrics["pending_str"], "0.00%")


if __name__ == "__main__":
    unittest.main()
