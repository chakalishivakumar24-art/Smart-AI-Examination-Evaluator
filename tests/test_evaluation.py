import unittest
import os
import tempfile
from pathlib import Path
from database.auth import hash_password, verify_password, authenticate_user
from models.database_models import StudentDetails, QuestionEvaluation, EvaluationRecord, ModelAnswerItem, SystemSettings
from ocr.first_page_ocr import extract_student_details_from_text
from ocr.answer_ocr import parse_questions_and_answers
from evaluation.similarity import compute_similarity
from evaluation.grading import evaluate_keyword_coverage, calculate_question_marks, determine_grade_and_status
from database.database import EvaluationDatabase
from reports.report_generator import generate_pdf_report
from reports.excel_csv_exporter import export_to_csv, export_to_excel

class TestPart2Features(unittest.TestCase):

    def test_model_answers_crud(self):
        db_file = os.path.join(os.path.dirname(__file__), "part2_crud_test.db")
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except Exception:
                pass

        try:
            db = EvaluationDatabase(db_path=db_file)
            item = ModelAnswerItem(
                subject_code="CS501", subject_name="Machine Learning",
                question_id="Q1", question_text="What is Supervised Learning?",
                model_answer="Supervised learning uses labeled datasets.", max_marks=10.0,
                keywords=["supervised", "labeled"]
            )
            db.add_model_answer(item)
            
            models = db.get_model_answers("CS501")
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0].question_id, "Q1")

            # Delete
            db.delete_model_answer(models[0].id)
            models_after = db.get_model_answers("CS501")
            self.assertEqual(len(models_after), 0)
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except Exception:
                    pass

    def test_dynamic_grading_and_pass_threshold(self):
        settings = SystemSettings(pass_percentage=50.0, grade_a_plus=90.0, grade_a=80.0, grade_e=50.0)

        # 85% score with custom settings
        grade85, status85 = determine_grade_and_status(85.0, settings)
        self.assertEqual(grade85, "A")
        self.assertEqual(status85, "PASS")

        # 45% score with pass threshold 50% -> should FAIL
        grade45, status45 = determine_grade_and_status(45.0, settings)
        self.assertEqual(status45, "FAIL")

    def test_excel_and_csv_exporters(self):
        student = StudentDetails(
            student_name="Rahul Kumar", roll_number="21CS045", registration_number="REG2026CS045",
            subject_name="Machine Learning", subject_code="CS501", exam_name="Semester Examination",
            exam_date="20-08-2026", college_name="ABC Engineering College", semester="V", branch="Computer Science"
        )
        rec = EvaluationRecord(
            booklet_filename="rahul.pdf", student=student, total_max_marks=10.0,
            ai_total_marks=8.0, final_total_marks=8.0, percentage=80.0, grade="A",
            is_approved=True, questions=[]
        )

        csv_bytes = export_to_csv([rec])
        self.assertIsNotNone(csv_bytes)
        self.assertIn(b"Rahul Kumar", csv_bytes)

        excel_bytes = export_to_excel([rec])
        self.assertIsNotNone(excel_bytes)
        self.assertGreater(len(excel_bytes), 200)

    def test_search_filter_and_sort_queries(self):
        db_file = os.path.join(os.path.dirname(__file__), "part2_search_test.db")
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except Exception:
                pass

        try:
            db = EvaluationDatabase(db_path=db_file)

            s1 = StudentDetails(student_name="Alice Smith", roll_number="101", subject_name="AI", exam_name="Final")
            r1 = EvaluationRecord(student=s1, total_max_marks=100, ai_total_marks=90, final_total_marks=90, percentage=90.0, grade="A+", is_approved=True)

            s2 = StudentDetails(student_name="Bob Jones", roll_number="102", subject_name="AI", exam_name="Final")
            r2 = EvaluationRecord(student=s2, total_max_marks=100, ai_total_marks=30, final_total_marks=30, percentage=30.0, grade="F", is_approved=True)

            db.save_evaluation(r1)
            db.save_evaluation(r2)

            # Search by name
            res_alice = db.search_evaluations(query="Alice")
            self.assertEqual(len(res_alice), 1)
            self.assertEqual(res_alice[0].student.student_name, "Alice Smith")

            # Filter by PASS
            res_pass = db.search_evaluations(status_filter="PASS")
            self.assertEqual(len(res_pass), 1)
            self.assertEqual(res_pass[0].student.student_name, "Alice Smith")

            # Filter by FAIL
            res_fail = db.search_evaluations(status_filter="FAIL")
            self.assertEqual(len(res_fail), 1)
            self.assertEqual(res_fail[0].student.student_name, "Bob Jones")
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except Exception:
                    pass

if __name__ == "__main__":
    unittest.main()
