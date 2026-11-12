import json
import os
from utils.constants import (
    STATUS_OK,
    STATUS_LOW_CONFIDENCE,
    STATUS_NEEDS_REVIEW,
    STATUS_NOT_DETECTED,
    STATUS_NOT_PRESENT
)

NORMALIZED_WIDTH = 1600
NORMALIZED_HEIGHT = 2200

# Canonical Value-Only ROIs (x, y, w, h) on 1600 x 2200 canvas
# Strictly excludes printed labels, colons, and dividing lines to avoid neighbor bleed-through
DEFAULT_VALUE_ROIS = {
    "booklet_serial_number": (1100, 60, 430, 115),
    "test_number": (360, 320, 280, 90),
    "exam_name": (650, 310, 750, 80),
    "enrollment_number": (1045, 360, 480, 75),
    "instructor_name": (1030, 440, 510, 85),
    "student_name": (185, 525, 435, 85),
    "section": (745, 530, 280, 95),
    "exam_date": (1145, 525, 395, 85),
    "class": (220, 605, 400, 85),
    "course_title": (600, 630, 950, 90),
    "semester": (220, 685, 400, 75),
    "course_code": (260, 755, 360, 75),
    "student_signature": (970, 685, 570, 75),
    "invigilator_signature": (980, 755, 560, 75),
    "additional_sheets": (380, 825, 240, 75),
    "examiner_remarks": (1120, 900, 420, 130),
    "recheck_request": (650, 900, 470, 130),
    "marks_table": (50, 980, 1500, 1170),
    "grand_total": (500, 1840, 240, 140)
}

# Printed Label Anchors (x, y, w, h) on 1600 x 2200 canvas
LABEL_ANCHORS = {
    "booklet_serial_number": (850, 70, 250, 90),
    "test_number": (260, 310, 100, 80),
    "enrollment_number": (740, 355, 280, 85),
    "instructor_name": (740, 440, 280, 85),
    "student_name": (50, 525, 135, 85),
    "section": (620, 525, 125, 85),
    "exam_date": (1020, 525, 125, 85),
    "class": (50, 605, 170, 85),
    "course_title": (620, 600, 230, 90),
    "semester": (50, 685, 170, 75),
    "course_code": (50, 755, 210, 75),
    "additional_sheets": (50, 825, 330, 75),
    "student_signature": (620, 685, 350, 75),
    "invigilator_signature": (620, 755, 360, 75),
    "grand_total": (140, 1940, 260, 80)
}

# Marks Table Canonical Grid Geometry on 1600 x 2200 canvas
MARKS_TABLE_COLUMNS = {
    "A": (145, 237),
    "B": (237, 329),
    "C": (329, 420),
    "D": (420, 512),
    "E": (512, 600),
    "Total": (600, 716)
}

MARKS_TABLE_ROWS = {
    "Q1": (1165, 1245),
    "Q2": (1245, 1325),
    "Q3": (1325, 1410),
    "Q4": (1410, 1495),
    "Q5": (1495, 1575),
    "Q6": (1575, 1660),
    "Q7": (1660, 1740),
    "Q8": (1740, 1815),
    "Q9": (1815, 1895),
    "Q10": (1895, 1970),
    "Grand_Total_Row": (1970, 2045)
}

DEFAULT_BOOKLET_TEMPLATE = DEFAULT_VALUE_ROIS.copy()

ICFAI_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "icfaitech_template.json")
EXAM_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "exam_template.json")

def load_template_config():
    for t_path in [ICFAI_TEMPLATE_PATH, EXAM_TEMPLATE_PATH]:
        if os.path.exists(t_path):
            try:
                with open(t_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    canvas = data.get("canvas", {})
                    w_c = canvas.get("width", 1600)
                    h_c = canvas.get("height", 2200)
                    fields = data.get("fields", {})
                    
                    loaded_rois = {}
                    for key, coords in fields.items():
                        x1 = int(coords.get("x1", 0) * w_c)
                        y1 = int(coords.get("y1", 0) * h_c)
                        x2 = int(coords.get("x2", 1) * w_c)
                        y2 = int(coords.get("y2", 1) * h_c)
                        loaded_rois[key] = (x1, y1, x2 - x1, y2 - y1)
                    return loaded_rois
            except Exception as e:
                print(f"[TEMPLATE CONFIG WARNING] Could not load JSON template ({t_path}): {e}")
    return DEFAULT_BOOKLET_TEMPLATE

BOOKLET_TEMPLATE = load_template_config()

MARKS_ROWS = 10
MARK_COLUMNS = ["A", "B", "C", "D", "E", "Total"]
