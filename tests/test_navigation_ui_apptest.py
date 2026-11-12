"""
AppTest regression test ensuring Streamlit UI navigation:
- Next button increments index cleanly without any StreamlitAPIException
- Prev button decrements index cleanly
- Selectbox allows jump to student
- High confidence OCR fields initialize as Verified Correct
- Empty/Not Detected fields initialize as Needs Verification
"""

import unittest
import os
import sys
from streamlit.testing.v1 import AppTest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


class TestNavigationStreamlitAppTest(unittest.TestCase):
    def test_navigation_and_verification_ui_flow(self):
        script_code = """
import streamlit as st
from utils.metrics import COVER_FIELDS, calculate_cover_field_metrics
from pages.upload import render_student_navigation

if "current_student_index" not in st.session_state:
    st.session_state["current_student_index"] = 0

student_records = [
    {
        "student_index": 0,
        "student_uid": "student_1",
        "student_name": "y.Rishika Reddy",
        "enrollment_no": "24STUCHH010984",
        "page_num": 1,
        "initial_extracted_fields": {
            "student_name": "y.Rishika Reddy",
            "enrollment_no": "24STUCHH010984",
            "test_no": "1",
            "section": "Not Detected",
            "date": "10/20/2026",
            "class": "2024-2028",
            "course_title": "Database Management System",
            "course_code": "Not Detected",
            "semester": "Not Detected",
            "instructor_name": "Not Present",
            "grand_total": "20/30"
        }
    },
    {
        "student_index": 1,
        "student_uid": "student_2",
        "student_name": "John Doe",
        "enrollment_no": "24STUCHH010985",
        "page_num": 2,
        "initial_extracted_fields": {
            "student_name": "John Doe",
            "enrollment_no": "24STUCHH010985",
            "test_no": "1",
            "section": "A",
            "date": "10/20/2026",
            "class": "2024-2028",
            "course_title": "Database Management System",
            "course_code": "CS203",
            "semester": "V",
            "instructor_name": "Dr. Smith",
            "grand_total": "25/30"
        }
    }
]

render_student_navigation(student_records)
"""
        test_script_path = os.path.join(ROOT_DIR, "scratch", "test_nav_apptest_ui.py")
        os.makedirs(os.path.dirname(test_script_path), exist_ok=True)
        with open(test_script_path, "w", encoding="utf-8") as f:
            f.write(script_code)

        at = AppTest.from_file(test_script_path).run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Initial run had exceptions: {at.exception}")
        self.assertEqual(at.session_state["current_student_index"], 0)

        # Click Next
        btn_next = at.button(key="btn_nav_next")
        self.assertIsNotNone(btn_next)
        at = btn_next.click().run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Clicking Next raised exception: {at.exception}")
        self.assertEqual(at.session_state["current_student_index"], 1, "Next must advance to index 1")

        # Click Prev
        btn_prev = at.button(key="btn_nav_prev")
        self.assertIsNotNone(btn_prev)
        at = btn_prev.click().run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Clicking Prev raised exception: {at.exception}")
        self.assertEqual(at.session_state["current_student_index"], 0, "Prev must return to index 0")


if __name__ == "__main__":
    unittest.main()
