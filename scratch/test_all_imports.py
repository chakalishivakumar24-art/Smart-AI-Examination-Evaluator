import sys
import os

print("--- TESTING ALL PAGE IMPORTS ---")
try:
    import database.database
    import models.database_models
    import template_config
    import utils.text_parser
    import ocr.preprocessing
    import ocr.student_details
    import ocr.marks_table
    import ocr.first_page_ocr
    import ocr.answer_ocr
    import evaluation.evaluator
    import pages.dashboard
    import pages.upload
    import pages.results
    import pages.student_details
    import pages.model_answers
    import pages.reports
    import pages.settings
    print("SUCCESS: All application modules imported cleanly with 0 syntax or import errors!")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
