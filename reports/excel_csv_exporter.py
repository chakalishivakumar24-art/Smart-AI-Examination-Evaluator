import csv
from io import StringIO, BytesIO
from typing import List
from models.database_models import EvaluationRecord, SystemSettings
from evaluation.grading import determine_grade_and_status

def build_rows_dict(records: List[EvaluationRecord], settings: SystemSettings = None) -> List[dict]:
    if settings is None:
        settings = SystemSettings()

    rows = []
    for idx, r in enumerate(records, 1):
        _, status = determine_grade_and_status(r.percentage, settings)
        rows.append({
            "S.No": idx,
            "Student Name": r.student.student_name,
            "Roll Number": r.student.roll_number,
            "Registration Number": r.student.registration_number,
            "Subject Name": r.student.subject_name,
            "Subject Code": r.student.subject_code,
            "Exam Name": r.student.exam_name,
            "Exam Date": r.student.exam_date,
            "College / University": r.student.college_name,
            "Semester": r.student.semester,
            "Branch": r.student.branch,
            "Total Marks": r.final_total_marks,
            "Maximum Marks": r.total_max_marks,
            "Percentage": f"{r.percentage}%",
            "Grade": r.grade,
            "Status": status,
            "Teacher Approved": "Yes" if r.is_approved else "No",
            "Evaluation Date": r.evaluation_date
        })
    return rows

def export_to_csv(records: List[EvaluationRecord], settings: SystemSettings = None) -> bytes:
    """Exports evaluation records to CSV byte stream."""
    rows = build_rows_dict(records, settings)
    if not rows:
        return b""

    fieldnames = list(rows[0].keys())
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode('utf-8')

def export_to_excel(records: List[EvaluationRecord], settings: SystemSettings = None) -> bytes:
    """Exports evaluation records to Microsoft Excel (.xlsx) byte stream using pandas or CSV fallback."""
    try:
        import pandas as pd
        df = pd.DataFrame(build_rows_dict(records, settings))
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Student Evaluation Results")
        return output.getvalue()
    except Exception:
        # Fallback to CSV bytes if pandas or openpyxl missing
        return export_to_csv(records, settings)
