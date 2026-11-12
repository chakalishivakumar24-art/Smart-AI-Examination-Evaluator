import unittest
from ocr.first_page_ocr import extract_student_details_from_text
from ocr.answer_ocr import parse_questions_and_answers
from utils.text_parser import clean_ocr_value

class TestOCRModule(unittest.TestCase):

    def test_clean_ocr_value_stripper(self):
        self.assertEqual(clean_ocr_value("Semester: IV", "semester"), "IV")
        self.assertEqual(clean_ocr_value("Course Code: CS501", "subject_code"), "CS501")
        self.assertEqual(clean_ocr_value("Subject: ()", "subject_name"), "")
        self.assertEqual(clean_ocr_value("Sem", "semester"), "")

    def test_first_page_ocr_metadata_regex(self):
        text = """
        STUDENT NAME: Rahul Kumar
        HALL TICKET NO: 21CS045
        REGISTRATION NO: REG2026CS045
        PAPER NAME: Machine Learning
        PAPER CODE: CS501
        EXAMINATION: Semester Examination
        DATE: 20-08-2026
        COLLEGE: ABC Engineering College
        SEMESTER: V
        BRANCH: Computer Science
        """
        details, confs = extract_student_details_from_text(text)
        self.assertEqual(details.student_name, "Rahul Kumar")
        self.assertEqual(details.roll_number, "21CS045")
        self.assertEqual(details.registration_number, "REG2026CS045")
        self.assertEqual(details.subject_name, "Machine Learning")
        self.assertEqual(details.subject_code, "CS501")

    def test_answer_page_question_parser(self):
        text = """
        Q1. Supervised learning requires labeled training data.
        Question 2: Overfitting occurs when the model memorizes training noise.
        3) Support Vector Machines maximize margin.
        """
        q_map = parse_questions_and_answers(text)
        self.assertIn("Q1", q_map)
        self.assertIn("Q2", q_map)
        self.assertIn("Q3", q_map)

if __name__ == "__main__":
    unittest.main()
