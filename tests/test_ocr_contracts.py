import unittest
from PIL import Image
from models.database_models import StudentDetails
from ocr.preprocessing import preprocess_image, preprocess_image_for_ocr
from ocr.first_page_ocr import (
    run_ocr,
    run_ocr_with_layout,
    extract_student_details,
    process_first_page,
    extract_first_page_metadata
)
from ocr.answer_ocr import (
    process_answer_page,
    process_answer_page_with_layout,
    extract_answers_from_pages
)

class TestOCRFunctionContracts(unittest.TestCase):

    def setUp(self):
        # Create a test dummy image
        self.test_img = Image.new('RGB', (800, 600), color=(255, 255, 255))

    def test_preprocess_image_contract(self):
        result = preprocess_image(self.test_img)
        self.assertIsInstance(result, Image.Image)

    def test_preprocess_image_for_ocr_contract(self):
        res = preprocess_image_for_ocr(self.test_img)
        self.assertEqual(len(res), 3)
        thresh, pil_img, angle = res
        self.assertIsInstance(pil_img, Image.Image)
        self.assertIsInstance(angle, int)

    def test_run_ocr_contract(self):
        res = run_ocr(self.test_img)
        self.assertEqual(len(res), 2)
        text, confidence = res
        self.assertIsInstance(text, str)
        self.assertIsInstance(confidence, float)

    def test_run_ocr_with_layout_contract(self):
        res = run_ocr_with_layout(self.test_img)
        self.assertEqual(len(res), 3)
        text, confidence, boxes = res
        self.assertIsInstance(text, str)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(boxes, list)

    def test_extract_student_details_contract(self):
        result = extract_student_details("STUDENT NAME: John Doe")
        self.assertIsInstance(result, StudentDetails)
        self.assertEqual(result.student_name, "John Doe")

    def test_process_first_page_contract(self):
        res = process_first_page(self.test_img)
        self.assertEqual(len(res), 3)
        student_details, ocr_text, confidence = res
        self.assertIsInstance(student_details, StudentDetails)
        self.assertIsInstance(ocr_text, str)
        self.assertIsInstance(confidence, float)

    def test_extract_first_page_metadata_contract(self):
        res = extract_first_page_metadata(self.test_img)
        self.assertEqual(len(res), 4)
        student_details, confidences, ocr_text, best_angle = res
        self.assertIsInstance(student_details, StudentDetails)
        self.assertIsInstance(confidences, dict)
        self.assertIsInstance(ocr_text, str)
        self.assertIsInstance(best_angle, int)

    def test_process_answer_page_contract(self):
        res = process_answer_page(self.test_img)
        self.assertEqual(len(res), 2)
        answer_text, confidence = res
        self.assertIsInstance(answer_text, str)
        self.assertIsInstance(confidence, float)

    def test_process_answer_page_with_layout_contract(self):
        res = process_answer_page_with_layout(self.test_img)
        self.assertEqual(len(res), 3)
        answer_text, confidence, boxes = res
        self.assertIsInstance(answer_text, str)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(boxes, list)

    def test_extract_answers_from_pages_contract(self):
        res = extract_answers_from_pages([self.test_img])
        self.assertEqual(len(res), 2)
        q_answers, full_text = res
        self.assertIsInstance(q_answers, dict)
        self.assertIsInstance(full_text, str)

if __name__ == "__main__":
    unittest.main()
