import unittest
from ocr.student_details import (
    normalize_enrollment_number, normalize_student_name, normalize_date
)

class TestOCRNormalizersDoNotInventValues(unittest.TestCase):
    def test_valid_enrollment_is_normalized_generically(self):
        self.assertEqual(normalize_enrollment_number("24STUCHH010015"), "24STUCHH010015")
        self.assertEqual(normalize_enrollment_number("23STUCHH010066"), "23STUCHH010066")

    def test_label_fragment_is_not_accepted_as_enrollment(self):
        self.assertEqual(normalize_enrollment_number("MINATION"), "")
        self.assertEqual(normalize_enrollment_number("24STUCHH0100"), "")

    def test_name_normalizer_does_not_replace_names_with_sample_names(self):
        self.assertEqual(normalize_student_name("Bhavana Nand"), "Bhavana Nand")
        self.assertEqual(normalize_student_name("J. Neethu"), "J. Neethu")

    def test_date_normalizer_does_not_guess_missing_digits(self):
        self.assertEqual(normalize_date("2/8/2026"), "02/08/2026")
        self.assertEqual(normalize_date("27/11/2025"), "27/11/2025")
        self.assertEqual(normalize_date("10/20/2026"), "20/10/2026")
        self.assertEqual(normalize_date("Date: 26-03-25"), "26/03/2025")
        self.assertEqual(normalize_date("Not Detected"), "")
        self.assertEqual(normalize_date("45/99/2025"), "")
        self.assertEqual(normalize_date(""), "")

if __name__ == "__main__":
    unittest.main()
