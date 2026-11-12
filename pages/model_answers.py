import streamlit as st
from database.database import EvaluationDatabase
from models.database_models import ModelAnswerItem

def render_model_answers_page(db: EvaluationDatabase):
    st.subheader("Reference Model Answers Key Manager (Full CRUD)")
    st.write("Add, edit, or delete reference model answers, maximum marks, and key concepts stored in SQLite.")

    subjects = db.get_all_subjects()
    subject_codes = [s["subject_code"] for s in subjects] if subjects else ["CS501"]
    
    selected_code = st.selectbox("Select Subject Code", subject_codes)
    models = db.get_model_answers(selected_code)

    st.markdown("---")
    st.write(f"##### Existing Model Answers for {selected_code}")

    if models:
        for m in models:
            with st.expander(f"📌 {m.question_id}: {m.question_text} (Max Marks: {m.max_marks})", expanded=True):
                c_info, c_del = st.columns([4, 1])
                with c_info:
                    st.write(f"**Reference Model Answer:** {m.model_answer}")
                    st.write(f"**Important Keywords:** {', '.join(m.keywords)}")
                with c_del:
                    if st.button(f"🗑️ Delete {m.question_id}", key=f"del_page_{m.id}"):
                        db.delete_model_answer(m.id)
                        st.success(f"Deleted {m.question_id}!")
                        st.rerun()
    else:
        st.info("No model answers configured for this subject code yet.")

    st.markdown("---")
    st.write("##### Add / Update Model Answer Key")
    with st.form("add_model_answer_form_page"):
        f_code = st.text_input("Subject Code", value=selected_code)
        f_name = st.text_input("Subject Name", value="Machine Learning")
        f_q_id = st.text_input("Question ID (e.g. Q1, Q2)", value="Q1")
        f_q_text = st.text_area("Question Text", value="Define Supervised Learning.")
        f_max = st.number_input("Maximum Marks", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
        f_ans = st.text_area("Reference Model Answer", value="Supervised learning trains models on labeled datasets...")
        f_kws = st.text_input("Keywords (comma separated)", value="supervised learning, labeled dataset, classification")

        submit_model = st.form_submit_button("Save Model Answer to Database", type="primary")

    if submit_model:
        kw_list = [k.strip() for k in f_kws.split(",") if k.strip()]
        new_item = ModelAnswerItem(
            subject_code=f_code, subject_name=f_name, question_id=f_q_id,
            question_text=f_q_text, model_answer=f_ans, max_marks=f_max, keywords=kw_list
        )
        db.add_model_answer(new_item)
        st.success(f"Saved model answer for {f_q_id}!")
        st.rerun()
