
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
