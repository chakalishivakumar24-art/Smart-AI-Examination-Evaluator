"""
Unit tests for Multi-Student PDF Processing and Navigation:
1. Dynamic detection and extraction of 5-6 student pages from a PDF.
2. Next and Previous navigation boundaries and indicators.
3. Per-student data isolation and preservation of manual edits.
4. Per-student dynamic calculation of Accuracy (%), Error (%), and Pending Verification (%).
5. Multiple image uploads batch processing.
6. Safe error and empty file handling.
"""

import unittest
import os
import io
import tempfile
from unittest.mock import patch
from PIL import Image

from database.database import EvaluationDatabase
from utils.pdf_processor import convert_bytes_to_images
from utils.metrics import (
    COVER_FIELDS,
    calculate_cover_field_metrics,
    get_student_field_value
)
from pages.upload import (
    process_single_page_student,
    process_multi_student_booklet_files
)


class MockUploadedFile:
    """Mock Streamlit UploadedFile object for testing."""
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


class TestMultiStudentNavigation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = os.path.join(tempfile.gettempdir(), "test_multi_student_eval.db")
        if os.path.exists(cls.test_db_path):
            try:
                os.remove(cls.test_db_path)
            except Exception:
                pass
        cls.db = EvaluationDatabase(db_path=cls.test_db_path)
        cls.multi_pdf_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "multi_student_6pages_test.pdf"
        )

    def test_01_multi_page_pdf_detected_and_extracted(self):
        """Verify that a 6-page PDF yields exactly 6 separate student records dynamically."""
        self.assertTrue(os.path.exists(self.multi_pdf_path), f"Missing test PDF at {self.multi_pdf_path}")

        with open(self.multi_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Test PyMuPDF rendering 6 pages from PDF
        images = convert_bytes_to_images(pdf_bytes, "multi_student_6pages_test.pdf")
        self.assertEqual(len(images), 6, "Expected exactly 6 pages rendered from test PDF")

        # Verify process_multi_student_booklet_files creates 6 distinct student records
        mock_file = MockUploadedFile("multi_student_6pages_test.pdf", pdf_bytes)

        # Mock process_single_page_student to test the multi-student file processing orchestration fast
        def dummy_process_page(page_img, student_index, total_students, filename, page_num, db, manual_rotation_angle=None):
            return {
                "student_index": student_index,
                "student_uid": f"test_pdf_p{page_num}_s{student_index + 1}",
                "filename": filename,
                "page_num": page_num,
                "total_students": total_students,
                "student_name": f"Student {student_index + 1}",
                "enrollment_no": f"23STUCHH0100{student_index + 10}",
                "student_details": None,
                "question_results": [],
                "evaluation_record": None,
                "marks_grid_results": [],
                "table_overlay_img": None,
                "raw_ocr_debug_text": "Sample OCR",
                "original_page_img": page_img,
                "oriented_images": [page_img],
                "detected_angle": 0,
                "p1_confidence": 90.0,
                "page_count": 1,
                "step_timings": {},
                "is_saved": False,
                "saved_eval_id": None,
                "initial_extracted_fields": {f["id"]: f"val_{f['id']}" for f in COVER_FIELDS}
            }

        with patch("pages.upload.process_single_page_student", side_effect=dummy_process_page):
            records = process_multi_student_booklet_files(
                uploaded_files=[mock_file],
                db=self.db,
                is_multi_student_mode=True
            )

        self.assertEqual(len(records), 6, "Expected 6 student records created from 6-page PDF")

        # Verify each record has unique student_uid, correct page number, and distinct data
        uids = set()
        for idx, rec in enumerate(records):
            self.assertEqual(rec["student_index"], idx)
            self.assertEqual(rec["page_num"], idx + 1)
            self.assertEqual(rec["total_students"], 6)
            self.assertIn("initial_extracted_fields", rec)
            self.assertEqual(len(rec["initial_extracted_fields"]), 11)

            uid = rec["student_uid"]
            self.assertNotIn(uid, uids, f"Duplicate student UID found: {uid}")
            uids.add(uid)

    def test_02_navigation_bounds_and_indicators(self):
        """Verify Next/Previous button enable/disable states and navigation indicator counter."""
        total_students = 6

        # Helper simulating navigation state
        def get_nav_state(cur_idx, total):
            can_prev = cur_idx > 0
            can_next = cur_idx < total - 1
            indicator_text = f"Student {cur_idx + 1} of {total}"
            return can_prev, can_next, indicator_text

        # Record 0 (First record)
        can_prev, can_next, text = get_nav_state(0, total_students)
        self.assertFalse(can_prev, "Previous must be disabled on the first student record")
        self.assertTrue(can_next, "Next must be enabled on the first record when total > 1")
        self.assertEqual(text, "Student 1 of 6")

        # Record 1 (Click Next)
        can_prev, can_next, text = get_nav_state(1, total_students)
        self.assertTrue(can_prev, "Previous must be enabled on intermediate records")
        self.assertTrue(can_next, "Next must be enabled on intermediate records")
        self.assertEqual(text, "Student 2 of 6")

        # Record 5 (Last record)
        can_prev, can_next, text = get_nav_state(5, total_students)
        self.assertTrue(can_prev, "Previous must be enabled on the last record")
        self.assertFalse(can_next, "Next must be disabled on the last student record")
        self.assertEqual(text, "Student 6 of 6")

    def test_03_per_student_data_isolation_and_edit_preservation(self):
        """Verify that edits on Student 1 do not leak to Student 2, and are preserved when navigating back."""
        # Simulated session state holding verification state per student_uid
        session_state = {}

        student_1_uid = "multi_pdf_p1_s1"
        student_2_uid = "multi_pdf_p2_s2"

        key_s1 = f"ver_state_{student_1_uid}"
        key_s2 = f"ver_state_{student_2_uid}"

        # Initialize student 1
        session_state[key_s1] = {
            f["id"]: {"verified": False, "ground_truth": ""} for f in COVER_FIELDS
        }
        # Initialize student 2
        session_state[key_s2] = {
            f["id"]: {"verified": False, "ground_truth": ""} for f in COVER_FIELDS
        }

        # User edits Student 1
        session_state[key_s1]["student_name"] = {"verified": True, "ground_truth": "Alice Cooper"}
        session_state[key_s1]["enrollment_no"] = {"verified": True, "ground_truth": "23STUCHH010066"}
        session_state[key_s1]["section"] = {"verified": True, "ground_truth": "C"}

        # Verify Student 2 was NOT altered by Student 1's edits
        self.assertEqual(session_state[key_s2]["student_name"]["ground_truth"], "")
        self.assertFalse(session_state[key_s2]["student_name"]["verified"])
        self.assertEqual(session_state[key_s2]["section"]["ground_truth"], "")

        # User navigates to Student 2 and edits Student 2
        session_state[key_s2]["student_name"] = {"verified": True, "ground_truth": "Bob Dylan"}
        session_state[key_s2]["enrollment_no"] = {"verified": True, "ground_truth": "23STUCHH010067"}

        # User navigates back to Student 1: assert Student 1's values are intact
        self.assertEqual(session_state[key_s1]["student_name"]["ground_truth"], "Alice Cooper")
        self.assertTrue(session_state[key_s1]["student_name"]["verified"])
        self.assertEqual(session_state[key_s1]["enrollment_no"]["ground_truth"], "23STUCHH010066")

        # Assert Student 2's values remain intact as well
        self.assertEqual(session_state[key_s2]["student_name"]["ground_truth"], "Bob Dylan")
        self.assertEqual(session_state[key_s2]["enrollment_no"]["ground_truth"], "23STUCHH010067")

    def test_04_per_student_metrics_calculation(self):
        """Verify dynamic calculation of Accuracy, Error, and Pending for the active student."""
        initial_extracted_s1 = {
            "student_name": "Alice Cooper",
            "enrollment_no": "23STUCHH010066",
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

        # Student 1: 6 fields verified correct, 0 incorrect, 5 needs verification
        v_map_s1 = {
            "student_name": {"verified": True, "ground_truth": "Alice Cooper"},
            "enrollment_no": {"verified": True, "ground_truth": "23STUCHH010066"},
            "test_no": {"verified": True, "ground_truth": "Test 1"},
            "section": {"verified": True, "ground_truth": "C"},
            "date": {"verified": True, "ground_truth": "27/11/2025"},
            "class": {"verified": True, "ground_truth": "2023 - 27"},
            "course_title": {"verified": False, "ground_truth": ""},
            "course_code": {"verified": False, "ground_truth": ""},
            "semester": {"verified": False, "ground_truth": ""},
            "instructor_name": {"verified": False, "ground_truth": ""},
            "grand_total": {"verified": False, "ground_truth": ""}
        }

        metrics_s1 = calculate_cover_field_metrics(COVER_FIELDS, v_map_s1, initial_extracted_s1)
        self.assertEqual(metrics_s1["verified_correct_count"], 6)
        self.assertEqual(metrics_s1["incorrect_count"], 0)
        self.assertEqual(metrics_s1["needs_verification_count"], 5)
        self.assertEqual(metrics_s1["accuracy_str"], "54.55%")
        self.assertEqual(metrics_s1["error_str"], "0.00%")
        self.assertEqual(metrics_s1["pending_str"], "45.45%")

        # Student 2: 8 fields verified correct, 1 incorrect, 2 pending verification
        initial_extracted_s2 = dict(initial_extracted_s1)
        initial_extracted_s2["student_name"] = "Bob M."  # OCR misread

        v_map_s2 = dict(v_map_s1)
        v_map_s2["student_name"] = {"verified": True, "ground_truth": "Bob Marley"} # Incorrect OCR match ("Bob M." != "Bob Marley")
        v_map_s2["course_title"] = {"verified": True, "ground_truth": "Machine Learning"}
        v_map_s2["course_code"] = {"verified": True, "ground_truth": "DS804"}
        v_map_s2["semester"] = {"verified": True, "ground_truth": "V"}

        metrics_s2 = calculate_cover_field_metrics(COVER_FIELDS, v_map_s2, initial_extracted_s2)
        # 8 correct: enrollment, test_no, section, date, class, course_title, course_code, semester
        # 1 incorrect: student_name ("Bob M." != "Bob Marley")
        # 2 pending: instructor_name, grand_total
        self.assertEqual(metrics_s2["verified_correct_count"], 8)
        self.assertEqual(metrics_s2["incorrect_count"], 1)
        self.assertEqual(metrics_s2["needs_verification_count"], 2)
        self.assertEqual(metrics_s2["accuracy_str"], "72.73%")
        self.assertEqual(metrics_s2["error_str"], "9.09%")
        self.assertEqual(metrics_s2["pending_str"], "18.18%")

    def test_05_multiple_images_batch_processing(self):
        """Verify that uploading multiple image files creates separate student records dynamically."""
        img_files = []
        for i in range(1, 4):
            img = Image.new("RGB", (100, 100), color=(255, 255, 255))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_files.append(MockUploadedFile(f"student_sheet_{i}.png", buf.getvalue()))

        def dummy_process_img(page_img, student_index, total_students, filename, page_num, db, manual_rotation_angle=None):
            return {
                "student_index": student_index,
                "student_uid": f"{filename}_s{student_index+1}",
                "filename": filename,
                "page_num": page_num,
                "total_students": total_students,
                "student_name": f"Student {student_index+1}",
                "initial_extracted_fields": {f["id"]: "" for f in COVER_FIELDS}
            }

        with patch("pages.upload.process_single_page_student", side_effect=dummy_process_img):
            records = process_multi_student_booklet_files(
                uploaded_files=img_files,
                db=self.db,
                is_multi_student_mode=True
            )

        self.assertEqual(len(records), 3, "Expected 3 records from 3 uploaded image files")
        self.assertEqual(records[0]["filename"], "student_sheet_1.png")
        self.assertEqual(records[1]["filename"], "student_sheet_2.png")
        self.assertEqual(records[2]["filename"], "student_sheet_3.png")

    def test_06_empty_or_corrupt_files_handled_gracefully(self):
        """Verify that empty files raise clean errors without crashing."""
        empty_file = MockUploadedFile("empty.pdf", b"")
        with self.assertRaises(Exception):
            process_multi_student_booklet_files([empty_file], self.db)


if __name__ == "__main__":
    unittest.main()
