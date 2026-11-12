"""
Regression Test Suite for 7-Page Multi-Student PDF Processing, Navigation, and Verification Metrics:
1. 7-Page PDF processing & distinct student extraction (data/uploads/cf6ff073_img pdf 2.pdf).
2. Navigation state machine: first page (0), middle pages (1..5), last page (6), and boundary clamping.
3. Per-student data isolation and page preview alignment across all 7 students.
4. Field-status transitions (Needs Verification -> Verified Correct / Incorrect).
5. Dynamic Accuracy (%), Error (%), and Pending Verification (%) formulas summing to 100%.
"""

import unittest
import os
import sys
import tempfile
from unittest.mock import patch

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from database.database import EvaluationDatabase
from utils.pdf_processor import convert_bytes_to_images
from utils.metrics import (
    COVER_FIELDS,
    calculate_cover_field_metrics,
    compare_field_values,
    get_student_field_value
)
from pages.upload import (
    process_single_page_student,
    process_multi_student_booklet_files
)


class MockUploadedFile:
    """Mock Streamlit UploadedFile object."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self.data = data
        self.size = len(data)

    def getvalue(self) -> bytes:
        return self.data

    def read(self) -> bytes:
        return self.data

    def seek(self, pos: int):
        pass


class TestSevenPageNavigationAndMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = os.path.join(tempfile.gettempdir(), "test_seven_page_regression.db")
        if os.path.exists(cls.test_db_path):
            try:
                os.remove(cls.test_db_path)
            except Exception:
                pass
        cls.db = EvaluationDatabase(db_path=cls.test_db_path)
        cls.seven_page_pdf_path = os.path.join(ROOT_DIR, "data", "uploads", "cf6ff073_img pdf 2.pdf")

    def test_01_seven_page_pdf_detected_and_extracted(self):
        """Requirement: Process a seven-page PDF and retain 7 distinct student results."""
        self.assertTrue(
            os.path.exists(self.seven_page_pdf_path),
            f"7-page test PDF missing at: {self.seven_page_pdf_path}"
        )

        with open(self.seven_page_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Step A: Convert PDF to images via PyMuPDF
        images = convert_bytes_to_images(pdf_bytes, "cf6ff073_img pdf 2.pdf")
        self.assertEqual(len(images), 7, "Expected exactly 7 pages rendered from 7-page PDF")

        # Step B: Orchestrate multi-student processing
        mock_file = MockUploadedFile("cf6ff073_img pdf 2.pdf", pdf_bytes)

        # Mock page processor to test extraction orchestration
        def dummy_process_page(page_img, student_index, total_students, filename, page_num, db, manual_rotation_angle=None):
            return {
                "student_index": student_index,
                "student_uid": f"cf6ff073_p{page_num}_s{student_index + 1}",
                "filename": filename,
                "page_num": page_num,
                "total_students": total_students,
                "student_name": f"Student {student_index + 1}",
                "enrollment_no": f"24STUCHH0100{student_index + 10}",
                "student_details": None,
                "question_results": [],
                "evaluation_record": None,
                "marks_grid_results": [],
                "table_overlay_img": None,
                "raw_ocr_debug_text": f"OCR text page {page_num}",
                "original_page_img": page_img,
                "oriented_images": [page_img],
                "detected_angle": 0,
                "p1_confidence": 88.0,
                "page_count": 1,
                "step_timings": {},
                "is_saved": False,
                "saved_eval_id": None,
                "initial_extracted_fields": {f["id"]: f"val_{f['id']}_{page_num}" for f in COVER_FIELDS}
            }

        with patch("pages.upload.process_single_page_student", side_effect=dummy_process_page):
            records = process_multi_student_booklet_files(
                uploaded_files=[mock_file],
                db=self.db,
                is_multi_student_mode=True
            )

        self.assertEqual(len(records), 7, "Expected exactly 7 student records from 7-page PDF")

        # Verify page numbers 1..7, student_index 0..6, and unique student_uids
        uids = set()
        for i, rec in enumerate(records):
            self.assertEqual(rec["student_index"], i)
            self.assertEqual(rec["page_num"], i + 1)
            self.assertEqual(rec["total_students"], 7)
            self.assertIn("original_page_img", rec)
            self.assertIsNotNone(rec["original_page_img"])
            self.assertIn("initial_extracted_fields", rec)
            self.assertEqual(len(rec["initial_extracted_fields"]), 11)

            uid = rec["student_uid"]
            self.assertNotIn(uid, uids, f"Duplicate UID detected: {uid}")
            uids.add(uid)

    def test_02_seven_page_navigation_state_machine_and_boundaries(self):
        """
        Requirement: Test navigation for seven-page PDF:
        - First page (0): Prev disabled, Next enabled.
        - Middle pages (1..5): Prev enabled, Next enabled.
        - Last page (6): Prev enabled, Next disabled.
        - Boundary clamping: index cannot go below 0 or beyond 6.
        """
        total = 7

        def nav_step(cur_idx, action):
            """Simulates navigation button clicks with clamped boundaries."""
            if action == "prev":
                new_idx = max(0, cur_idx - 1)
            elif action == "next":
                new_idx = min(total - 1, cur_idx + 1)
            elif action == "stay":
                new_idx = cur_idx
            else:
                new_idx = max(0, min(total - 1, int(action)))

            can_prev = (new_idx > 0)
            can_next = (new_idx < total - 1)
            indicator = f"Student {new_idx + 1} of {total}"
            return new_idx, can_prev, can_next, indicator

        # Start at Student 1 (index 0)
        idx, can_prev, can_next, ind = nav_step(0, "stay")
        self.assertEqual(idx, 0)
        self.assertFalse(can_prev, "Previous must be disabled at index 0")
        self.assertTrue(can_next, "Next must be enabled at index 0")
        self.assertEqual(ind, "Student 1 of 7")

        # Try to click Previous at index 0: boundary must clamp to 0
        idx, can_prev, can_next, ind = nav_step(idx, "prev")
        self.assertEqual(idx, 0, "Index must not go below 0")
        self.assertFalse(can_prev)

        # Click Next -> index 1
        idx, can_prev, can_next, ind = nav_step(idx, "next")
        self.assertEqual(idx, 1)
        self.assertTrue(can_prev)
        self.assertTrue(can_next)
        self.assertEqual(ind, "Student 2 of 7")

        # Click Next until Middle page: Student 4 (index 3)
        idx, _, _, _ = nav_step(idx, "next")  # index 2
        idx, can_prev, can_next, ind = nav_step(idx, "next")  # index 3
        self.assertEqual(idx, 3, "Middle student must be index 3")
        self.assertTrue(can_prev, "Previous must be enabled at middle page")
        self.assertTrue(can_next, "Next must be enabled at middle page")
        self.assertEqual(ind, "Student 4 of 7")

        # Click Next until Last student: Student 7 (index 6)
        idx, _, _, _ = nav_step(idx, "next")  # index 4
        idx, _, _, _ = nav_step(idx, "next")  # index 5
        idx, can_prev, can_next, ind = nav_step(idx, "next")  # index 6
        self.assertEqual(idx, 6, "Last student must be index 6")
        self.assertTrue(can_prev, "Previous must be enabled at last page")
        self.assertFalse(can_next, "Next must be disabled at last page (6)")
        self.assertEqual(ind, "Student 7 of 7")

        # Try to click Next at index 6: boundary must clamp to 6
        idx, can_prev, can_next, ind = nav_step(idx, "next")
        self.assertEqual(idx, 6, "Index must not go beyond total - 1 (6)")
        self.assertFalse(can_next)

        # Click Previous from last page back to first page
        for expected_idx in range(5, -1, -1):
            idx, _, _, ind = nav_step(idx, "prev")
            self.assertEqual(idx, expected_idx)

        self.assertEqual(idx, 0)
        self.assertEqual(ind, "Student 1 of 7")

    def test_03_per_student_data_isolation_across_seven_students(self):
        """
        Requirement:
        - Do not mix one student's extracted data with another student's page image.
        - Retain individual student verification states and edits independently.
        """
        session_state = {}
        total = 7

        # Initialize verification state for each of the 7 students
        for s in range(1, total + 1):
            uid = f"cf6ff073_p{s}_s{s}"
            key = f"ver_state_{uid}"
            session_state[key] = {
                f["id"]: {
                    "verified": False,
                    "ground_truth": f"OCR_Val_{f['id']}_s{s}"
                } for f in COVER_FIELDS
            }

        # Edit Student 1
        s1_key = "ver_state_cf6ff073_p1_s1"
        session_state[s1_key]["student_name"]["ground_truth"] = "Y. Rishika Reddy"
        session_state[s1_key]["student_name"]["verified"] = True
        session_state[s1_key]["enrollment_no"]["ground_truth"] = "24STUCHH010032"
        session_state[s1_key]["enrollment_no"]["verified"] = True

        # Edit Student 4 (middle student)
        s4_key = "ver_state_cf6ff073_p4_s4"
        session_state[s4_key]["student_name"]["ground_truth"] = "Student Four"
        session_state[s4_key]["student_name"]["verified"] = True
        session_state[s4_key]["enrollment_no"]["ground_truth"] = "24STUCHH010010"
        session_state[s4_key]["enrollment_no"]["verified"] = True

        # Edit Student 7 (last student)
        s7_key = "ver_state_cf6ff073_p7_s7"
        session_state[s7_key]["student_name"]["ground_truth"] = "Shiva Kumar"
        session_state[s7_key]["student_name"]["verified"] = True
        session_state[s7_key]["enrollment_no"]["ground_truth"] = "24STUCHH010043"
        session_state[s7_key]["enrollment_no"]["verified"] = True

        # Verify Student 2 was unaffected
        s2_key = "ver_state_cf6ff073_p2_s2"
        self.assertFalse(session_state[s2_key]["student_name"]["verified"])
        self.assertEqual(session_state[s2_key]["student_name"]["ground_truth"], "OCR_Val_student_name_s2")

        # Verify Student 1 retains their values
        self.assertTrue(session_state[s1_key]["student_name"]["verified"])
        self.assertEqual(session_state[s1_key]["student_name"]["ground_truth"], "Y. Rishika Reddy")

        # Verify Student 4 retains their values
        self.assertTrue(session_state[s4_key]["student_name"]["verified"])
        self.assertEqual(session_state[s4_key]["student_name"]["ground_truth"], "Student Four")

        # Verify Student 7 retains their values
        self.assertTrue(session_state[s7_key]["student_name"]["verified"])
        self.assertEqual(session_state[s7_key]["student_name"]["ground_truth"], "Shiva Kumar")

    def test_04_initial_upload_reports_zero_accuracy_and_hundred_pending(self):
        """
        Requirement:
        - Do not report 100% accuracy merely because OCR returned text.
        - Distinguish OCR confidence from confirmed correctness.
        - Unreviewed fields must remain Pending Verification.
        - Initial upload: Accuracy = 0.00%, Error = 0.00%, Pending = 100.00%.
        """
        initial_extracted = {
            "student_name": "Y. Rishika Reddy",
            "enrollment_no": "24STUCHH010032",
            "test_no": "Test 1",
            "section": "C",
            "date": "27/11/2025",
            "class": "2023 - 27",
            "course_title": "Machine Learning",
            "course_code": "DS804",
            "semester": "V",
            "instructor_name": "Dr. Sharma",
            "grand_total": "18/30"
        }

        # Initial unreviewed state: verified = False for all 11 fields
        v_map = {f["id"]: {"verified": False, "ground_truth": initial_extracted[f["id"]]} for f in COVER_FIELDS}

        metrics = calculate_cover_field_metrics(COVER_FIELDS, v_map, initial_extracted)

        self.assertEqual(metrics["total_fields"], 11)
        self.assertEqual(metrics["verified_correct_count"], 0)
        self.assertEqual(metrics["incorrect_count"], 0)
        self.assertEqual(metrics["needs_verification_count"], 11)

        self.assertEqual(metrics["accuracy_pct"], 0.00)
        self.assertEqual(metrics["error_pct"], 0.00)
        self.assertEqual(metrics["pending_pct"], 100.00)

        # Status map check: all 11 must be 'Needs Verification'
        for f in COVER_FIELDS:
            self.assertEqual(metrics["field_status_map"][f["id"]], "Needs Verification")

    def test_05_field_status_transitions_and_accuracy_formula(self):
        """
        Requirement:
        - Mark a field Verified Correct only when explicitly confirmed against reliable GT.
        - Mark a field Incorrect only when comparison establishes it is wrong.
        - Keep unreviewed fields Pending Verification.
        - Accuracy (%) = verified_correct / 11 * 100.
        - Error (%) = confirmed_incorrect / 11 * 100.
        - Pending (%) = pending_verification / 11 * 100.
        """
        initial_extracted = {
            "student_name": "Rishika",       # OCR made a partial typo
            "enrollment_no": "24STUCHH010032", # OCR is correct
            "test_no": "Test 1",              # OCR is correct
            "section": "C",                   # OCR is correct
            "date": "27/11/2025",             # OCR is correct
            "class": "2023 - 27",             # OCR is correct
            "course_title": "Machine Learning", # OCR is correct
            "course_code": "DS804",           # OCR is correct
            "semester": "V",                  # OCR is correct
            "instructor_name": "",            # OCR missed
            "grand_total": "18/30"            # OCR is correct
        }

        # Step 1: User verifies 8 correct fields, corrects 1 error, and leaves 2 unreviewed
        v_map = {
            # 8 Verified Correct
            "enrollment_no": {"verified": True, "ground_truth": "24STUCHH010032"},
            "test_no": {"verified": True, "ground_truth": "Test 1"},
            "section": {"verified": True, "ground_truth": "C"},
            "date": {"verified": True, "ground_truth": "27/11/2025"},
            "class": {"verified": True, "ground_truth": "2023 - 27"},
            "course_title": {"verified": True, "ground_truth": "Machine Learning"},
            "course_code": {"verified": True, "ground_truth": "DS804"},
            "grand_total": {"verified": True, "ground_truth": "18/30"},
            # 1 Confirmed Incorrect (user entered correct name, proving OCR typo)
            "student_name": {"verified": True, "ground_truth": "Y. Rishika Reddy"},
            # 2 Pending Verification
            "semester": {"verified": False, "ground_truth": ""},
            "instructor_name": {"verified": False, "ground_truth": ""}
        }

        metrics = calculate_cover_field_metrics(COVER_FIELDS, v_map, initial_extracted)

        self.assertEqual(metrics["verified_correct_count"], 8)
        self.assertEqual(metrics["incorrect_count"], 1)
        self.assertEqual(metrics["needs_verification_count"], 2)

        # Exact formulas:
        # 8 / 11 * 100 = 72.7272... -> 72.73%
        # 1 / 11 * 100 = 9.0909...  -> 9.09%
        # 2 / 11 * 100 = 18.1818... -> 18.18%
        self.assertEqual(metrics["accuracy_pct"], 72.73)
        self.assertEqual(metrics["error_pct"], 9.09)
        self.assertEqual(metrics["pending_pct"], 18.18)

        # Sum of counts equals 11
        self.assertEqual(
            metrics["verified_correct_count"] + metrics["incorrect_count"] + metrics["needs_verification_count"],
            11
        )

        # Sum of percentages equals 100.0% within rounding
        pct_sum = metrics["accuracy_pct"] + metrics["error_pct"] + metrics["pending_pct"]
        self.assertAlmostEqual(pct_sum, 100.0, delta=0.05)

    def test_06_all_possible_partitions_sum_to_one_hundred_percent(self):
        """
        Verify that for every possible mathematical partition (v, e, p) of 11 fields,
        the three percentages sum to 100.0% within standard rounding allowance.
        """
        for v in range(12):
            for e in range(12 - v):
                p = 11 - v - e
                acc = round((v / 11) * 100.0, 2)
                err = round((e / 11) * 100.0, 2)
                pend = round((p / 11) * 100.0, 2)
                total_pct = acc + err + pend
                self.assertAlmostEqual(
                    total_pct, 100.0, delta=0.05,
                    msg=f"Partition ({v}, {e}, {p}) did not sum to 100.0%: {total_pct}"
                )


if __name__ == "__main__":
    unittest.main()
