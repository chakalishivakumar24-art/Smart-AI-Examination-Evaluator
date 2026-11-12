import unittest
from evaluation.grading import evaluate_keyword_coverage, calculate_question_marks, determine_grade_and_status
from models.database_models import SystemSettings

class TestGradingModule(unittest.TestCase):

    def test_grade_bounds(self):
        settings = SystemSettings(pass_percentage=40.0)
        grade_ap, status_ap = determine_grade_and_status(92.0, settings)
        self.assertEqual(grade_ap, "A+")
        self.assertEqual(status_ap, "PASS")

        grade_f, status_f = determine_grade_and_status(35.0, settings)
        self.assertEqual(grade_f, "F")
        self.assertEqual(status_f, "FAIL")

    def test_keyword_matching(self):
        s_ans = "Supervised learning trains on labeled data for classification and regression."
        m_ans = "Supervised learning uses labeled dataset to predict discrete or continuous values."
        kws = ["supervised learning", "labeled data", "classification", "regression"]
        
        matched, missing, ratio = evaluate_keyword_coverage(s_ans, m_ans, kws)
        self.assertGreaterEqual(len(matched), 3)

        marks = calculate_question_marks(0.85, ratio, 1.0, 10.0)
        self.assertGreaterEqual(marks, 7.0)

if __name__ == "__main__":
    unittest.main()
