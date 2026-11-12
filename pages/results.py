import streamlit as st
import pandas as pd
from database.database import EvaluationDatabase
from evaluation.grading import determine_grade_and_status

def render_results_page(db: EvaluationDatabase):
    st.subheader("All Students Evaluation Results Matrix")
    st.write("Dynamic view of all evaluated student booklet records stored in SQLite database.")

    sys_settings = db.get_settings()
    evals = db.search_evaluations()

    if not evals:
        st.info("No student evaluation records found in database.")
    else:
        table_rows = []
        for idx, r in enumerate(evals, 1):
            g_val, s_val = determine_grade_and_status(r.percentage, sys_settings)
            table_rows.append({
                "S.No": idx,
                "Student Name": r.student.student_name,
                "Roll Number": r.student.roll_number,
                "Subject": f"{r.student.subject_name} ({r.student.subject_code})",
                "Exam": r.student.exam_name,
                "Total Marks": f"{r.final_total_marks} / {r.total_max_marks}",
                "Percentage": f"{r.percentage}%",
                "Grade": g_val,
                "Status": s_val,
                "Approved": "Yes" if r.is_approved else "Pending"
            })
        st.dataframe(pd.DataFrame(table_rows), width="stretch")

        st.markdown("---")
        st.write("##### Detailed Student Inspection Drawer")
        selected_st = st.selectbox("Select Student to View Complete Report", [f"#{r.id} - {r.student.student_name} ({r.student.roll_number})" for r in evals])
        st_id = int(selected_st.split(" - ")[0].replace("#", ""))

        target_rec = db.get_evaluation(st_id)
        if target_rec:
            st.write(f"### Student Evaluation Sheet: {target_rec.student.student_name}")
            s = target_rec.student
            g_val, s_val = determine_grade_and_status(target_rec.percentage, sys_settings)

            st.write("###### Marks Summary")
            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("Total Marks", target_rec.final_total_marks)
            sm2.metric("Max Marks", target_rec.total_max_marks)
            sm3.metric("Percentage", f"{target_rec.percentage}%")
            sm4.metric("Grade", g_val)
            sm5.metric("Status", s_val)

            st.write("###### Question-Wise Details")
            q_rows = []
            for q in target_rec.questions:
                q_rows.append({
                    "Question": q.question_id,
                    "Student Answer": q.student_answer,
                    "Model Answer": q.model_answer,
                    "Max Marks": q.max_marks,
                    "AI Suggested Marks": q.ai_marks,
                    "Teacher Final Marks": q.final_marks,
                    "Feedback": q.feedback,
                    "Teacher Comments": q.teacher_comments
                })
            st.dataframe(pd.DataFrame(q_rows), width="stretch")
