import streamlit as st
import pandas as pd
from database.database import EvaluationDatabase
from evaluation.grading import determine_grade_and_status

def render_student_details_page(db: EvaluationDatabase):
    st.subheader("Multi-Criteria Student Directory Search & Filter Engine")
    sys_settings = db.get_settings()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        s_query = st.text_input("🔍 Search Name / Roll / Reg", "")
    with c2:
        s_subj = st.selectbox("Filter Subject", ["All"] + [s["subject_name"] for s in db.get_all_subjects()])
    with c3:
        s_exam = st.selectbox("Filter Exam", ["All"] + db.get_all_exams())
    with c4:
        s_grade = st.selectbox("Filter Grade", ["All", "A+", "A", "B", "C", "D", "E", "F"])
    with c5:
        s_status = st.selectbox("Filter Status", ["All", "PASS", "FAIL"])

    sort_choice = st.selectbox("Sort Results By", ["Date", "Name", "Marks", "Percentage", "Grade"])

    filtered_records = db.search_evaluations(
        query=s_query, subject_filter=s_subj, exam_filter=s_exam,
        grade_filter=s_grade, status_filter=s_status, sort_by=sort_choice
    )

    if filtered_records:
        f_rows = []
        for idx, r in enumerate(filtered_records, 1):
            g_val, s_val = determine_grade_and_status(r.percentage, sys_settings)
            f_rows.append({
                "S.No": idx,
                "Student Name": r.student.student_name,
                "Roll Number": r.student.roll_number,
                "Registration No": r.student.registration_number,
                "Subject": r.student.subject_name,
                "Exam": r.student.exam_name,
                "Marks": f"{r.final_total_marks} / {r.total_max_marks}",
                "Percentage": f"{r.percentage}%",
                "Grade": g_val,
                "Status": s_val,
                "Date": r.evaluation_date[:10]
            })
        st.dataframe(pd.DataFrame(f_rows), width="stretch")
    else:
        st.warning("No student records match the specified search/filter criteria.")
