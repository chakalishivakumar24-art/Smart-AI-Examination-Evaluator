import sys
import textwrap

# Top-level direct execution protection to prevent ModuleNotFoundError when user runs "python app.py" directly
try:
    import streamlit as st
except ImportError:
    print("\n==================================================================")
    print(" [ERROR] Streamlit is not installed in your current Python environment.")
    print("------------------------------------------------------------------")
    print(" DO NOT RUN: python app.py")
    print(" CORRECT RUN: python -m streamlit run app.py")
    print("           OR: .venv\\Scripts\\python.exe -m streamlit run app.py")
    print("           OR: Double-click 'run_app.bat'")
    print("==================================================================\n")
    sys.exit(1)

import pandas as pd
import plotly.express as px
import os
from PIL import Image

# Import database & models
from database.auth import authenticate_user
from database.database import EvaluationDatabase
from models.database_models import StudentDetails, QuestionEvaluation, EvaluationRecord, ModelAnswerItem, SystemSettings

# Import pages
from pages.dashboard import render_dashboard_page
from pages.upload import render_upload_page
from pages.model_answers import render_model_answers_page
from pages.results import render_results_page
from pages.student_details import render_student_details_page
from pages.reports import render_reports_page
from pages.settings import render_settings_page

# Import utilities
from utils.pdf_processor import convert_bytes_to_images, split_booklet_pages, save_uploaded_file
from ocr.first_page_ocr import extract_first_page_metadata
from ocr.answer_ocr import extract_answers_from_pages
from evaluation.evaluator import evaluate_student_booklet
from evaluation.grading import determine_grade_and_status
from reports.report_generator import generate_pdf_report
from reports.excel_csv_exporter import export_to_csv, export_to_excel
from utils.sample_generator import seed_demo_evaluations, seed_sample_model_answers

def render_html(html_str: str):
    """
    Safely render HTML without markdown code-block conversion.
    Strips leading whitespace from each line so Markdown never treats 4-space indented HTML as code blocks.
    """
    clean_lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    st.markdown("\n".join(clean_lines), unsafe_allow_html=True)

# Page Config - Wide Layout
st.set_page_config(
    page_title="SMART AI EXAMINATION EVALUATOR",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean White Theme Custom CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Font Size Adjustments */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 16px !important;
    }
    
    /* Table & Dataframe Font Sizes */
    div[data-testid="stDataFrame"], div[data-testid="stTable"], table, .stDataFrame {
        font-size: 16px !important;
    }
    th {
        font-size: 16.5px !important;
        font-weight: 700 !important;
    }
    td {
        font-size: 16px !important;
    }

    /* Light Theme Input & Select Styling */
    div[data-baseweb="input"] input, div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-radius: 10px !important;
        border: 1px solid #DCE4EF !important;
        height: 54px !important;
        font-size: 17px !important;
    }
    
    div[data-baseweb="input"] input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12) !important;
    }

    /* Custom Icons inside Login Inputs */
    div[data-testid="stForm"] div[data-testid="stTextInput"]:nth-of-type(1) input {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/%3E%3Ccircle cx='12' cy='7' r='4'/%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-position: 14px center !important;
        background-size: 18px 18px !important;
        padding-left: 44px !important;
    }

    div[data-testid="stForm"] div[data-testid="stTextInput"]:nth-of-type(2) input {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='11' width='18' height='11' rx='2' ry='2'/%3E%3Cpath d='M7 11V7a5 5 0 0 1 10 0v4'/%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-position: 14px center !important;
        background-size: 18px 18px !important;
        padding-left: 44px !important;
    }

    /* Input Labels */
    .stTextInput label, .stSelectbox label, .stCheckbox label {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 17px !important;
        margin-bottom: 6px !important;
    }
    
    /* Hide default form border */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }

    /* Primary & Form Submit Button Styling ("Log In") */
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button {
        background: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        border-radius: 10px !important;
        border: none !important;
        height: 54px !important;
        padding: 0 24px !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25) !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
    }

    div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button:hover {
        background: #1D4ED8 !important;
        box-shadow: 0 6px 20px rgba(29, 78, 216, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    div.stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 17px !important;
        border-radius: 10px !important;
        border: 1px solid #DCE4EF !important;
        height: 52px !important;
        padding: 10px 18px !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        background: #F8FAFC !important;
        color: #0F172A !important;
        border-color: #CBD5E1 !important;
    }

    /* Checkbox & Text Adjustments */
    div[data-testid="stCheckbox"] label span {
        font-size: 16px !important;
        color: #475569 !important;
    }

    .pill-matched {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11.5px;
        font-weight: 600;
        margin: 2px;
        display: inline-block;
    }
    .pill-missing {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11.5px;
        font-weight: 600;
        margin: 2px;
        display: inline-block;
    }
    .ai-warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        color: #92400E;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database
@st.cache_resource
def get_db():
    return EvaluationDatabase()

db = get_db()
sys_settings = db.get_settings()

# Session State Initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "📊 Dashboard"
if "current_booklet_filename" not in st.session_state:
    st.session_state.current_booklet_filename = None
if "extracted_student" not in st.session_state:
    st.session_state.extracted_student = StudentDetails()
if "extracted_answers" not in st.session_state:
    st.session_state.extracted_answers = {}
if "booklet_images" not in st.session_state:
    st.session_state.booklet_images = []

# ==========================================
# AUTHENTICATION / LOGIN SCREEN (CLEAN)
# ==========================================
if not st.session_state.authenticated:
    _, col_center, _ = st.columns([1, 1.4, 1])

    with col_center:
        render_html("""
        <div style="text-align: center; background: #0F172A; border: 1px solid #1E293B; border-radius: 18px; padding: 28px 24px; margin-top: 20px; margin-bottom: 20px; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.15);">
            <div style="display: inline-flex; align-items: center; justify-content: center; width: 60px; height: 60px; background: rgba(59, 130, 246, 0.15); border: 1px solid #3B82F6; border-radius: 16px; font-size: 32px; margin-bottom: 12px;">🎓</div>
            <h1 style="font-size: 28px; font-weight: 900; margin: 0 0 6px 0; letter-spacing: -0.5px; line-height: 1.2;">
                <span style="color: #FFFFFF;">SMART AI</span> <span style="color: #3B82F6;">EXAMINATION</span> <span style="color: #FFFFFF;">EVALUATOR</span>
            </h1>
            <p style="font-size: 14.5px; color: #94A3B8; margin: 0; font-weight: 500;">Academic Teacher Portal Access</p>
        </div>
        """)

        with st.form("login_form"):
            username_val = st.text_input(
                "Teacher Name / Username", 
                placeholder="Enter email or username", 
                value="teacher@smartai.edu"
            )
            
            show_pwd = st.checkbox("Show password", value=False)
            
            password_val = st.text_input(
                "Password", 
                placeholder="Enter password", 
                type="default" if show_pwd else "password",
                value="teacher123"
            )
            
            col_rem, col_fg = st.columns([1.1, 0.9])
            with col_rem:
                remember_me = st.checkbox("Remember me", value=True)
            with col_fg:
                render_html("""
                <div style="text-align: right; margin-top: 6px;">
                    <a href="#forgot" style="color: #2563EB; font-size: 14px; font-weight: 600; text-decoration: none;" onclick="alert('Please contact system administrator to reset your credentials.'); return false;">Forgot password?</a>
                </div>
                """)

            login_submitted = st.form_submit_button("Sign In", type="primary")

            if login_submitted:
                with st.spinner("Signing in..."):
                    auth_user = authenticate_user(username_val, password_val)
                    
                    if auth_user:
                        st.session_state.authenticated = True
                        st.session_state.user_info = auth_user
                        st.session_state.current_page = "📊 Dashboard"
                        st.rerun()
                    else:
                        st.error("Invalid username or password. Please try again.")

    st.stop()

# ==========================================
# MAIN APP PORTAL WORKSPACE (TOP NAVIGATION)
# ==========================================

# 1. Top Header Bar
user_name = st.session_state.user_info['name'] if st.session_state.user_info and 'name' in st.session_state.user_info else "User"
user_role = st.session_state.user_info['role'] if st.session_state.user_info and 'role' in st.session_state.user_info else "Teacher"
user_initial = user_name[0] if user_name else "U"

render_html(f"""
<div style="background: #0F172A; border: 1px solid #1E293B; padding: 20px 28px; border-radius: 16px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);">
    <div style="display: flex; align-items: center; gap: 18px;">
        <div style="width: 52px; height: 52px; background: rgba(59, 130, 246, 0.15); border: 1px solid #3B82F6; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 28px;">🎓</div>
        <div>
            <h2 style="margin: 0; font-size: 26px; font-weight: 900; letter-spacing: -0.5px; line-height: 1.2;">
                <span style="color: #FFFFFF;">SMART AI</span> <span style="color: #3B82F6;">EXAMINATION</span> <span style="color: #FFFFFF;">EVALUATOR</span>
            </h2>
            <p style="margin: 4px 0 0 0; font-size: 14.5px; color: #94A3B8;">Teacher Portal Workspace | Evaluation Mode: <b style="color: #60A5FA;">{sys_settings.ai_evaluation_mode}</b> (Pass: <b style="color: #34D399;">{sys_settings.pass_percentage}%</b>)</p>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 14px; background: #1E293B; padding: 10px 20px; border-radius: 12px; border: 1px solid #334155;">
        <div style="width: 40px; height: 40px; background: #3B82F6; color: #FFFFFF; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 17px;">
            {user_initial}
        </div>
        <div style="text-align: left;">
            <div style="font-size: 15px; font-weight: 700; color: #FFFFFF;">{user_name}</div>
            <div style="font-size: 12px; color: #94A3B8; font-weight: 500;">{user_role}</div>
        </div>
    </div>
</div>
""")

# 2. Horizontal Top Navigation Bar Buttons
nav_items = [
    ("📊 Dashboard", "📊 Dashboard"),
    ("📤 Upload", "📤 Upload Answer Booklet"),
    ("📝 Model Answers", "📝 Model Answers"),
    ("⚖️ Evaluation", "⚖️ Answer Evaluation"),
    ("📈 Results", "📈 All Students Results"),
    ("🔍 Search", "🔍 Student Search & Filters"),
    ("📄 Reports", "📄 Reports & Exports"),
    ("⚙️ Settings", "⚙️ Settings"),
    ("🚪 Logout", "🚪 Logout")
]

nav_cols = st.columns(len(nav_items))

for idx, (label, full_name) in enumerate(nav_items):
    with nav_cols[idx]:
        is_active = (st.session_state.current_page == full_name)
        if st.button(label, key=f"topnav_{idx}", type="primary" if is_active else "secondary"):
            if full_name == "🚪 Logout":
                st.session_state.authenticated = False
                st.session_state.user_info = None
                st.session_state.extracted_student = StudentDetails()
                st.session_state.extracted_answers = {}
                st.rerun()
            else:
                st.session_state.current_page = full_name
                st.rerun()

st.markdown("<hr style='margin: 14px 0 24px 0; border: 0; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

# 3. Main Workspace Page Routing
nav_choice = st.session_state.current_page

if nav_choice == "📊 Dashboard":
    render_dashboard_page(db)

elif nav_choice == "📤 Upload Answer Booklet":
    render_upload_page(db)

elif nav_choice == "📝 Model Answers":
    render_model_answers_page(db)

elif nav_choice == "⚖️ Answer Evaluation":
    # Render Teacher Review workspace
    st.subheader("Teacher Answer Review & Score Approval Workspace")
    eval_list = db.search_evaluations()

    if not eval_list:
        st.info("No student booklet evaluations available to review. Upload a booklet or load demo data in Settings.")
    else:
        st.markdown("""
        <div class="ai-warning-box">
            ⚠️ <b>AI Warning Notice:</b> AI-generated marks are suggestions based on semantic similarity and keyword coverage. Please verify the student answer and approve the final marks.
        </div>
        """, unsafe_allow_html=True)

        eval_options = {f"#{e.id} - {e.student.student_name} ({e.student.roll_number}) - {e.student.subject_code}": e.id for e in eval_list}
        selected_opt = st.selectbox("Select Student Booklet to Review", list(eval_options.keys()))
        target_id = eval_options[selected_opt]

        rec = db.get_evaluation(target_id)

        if rec:
            s = rec.student
            grade_val, status_val = determine_grade_and_status(rec.percentage, sys_settings)

            st.markdown(f"""
            <div style="background:#F0F9FF; border:1px solid #BAE6FD; padding:16px; border-radius:10px; margin-bottom:18px;">
                <h4 style="margin:0 0 10px 0; color:#0369A1;">Student Information Details</h4>
                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; font-size:13px; color:#0F172A;">
                    <div><b>Name:</b> {s.student_name}</div>
                    <div><b>Roll No:</b> {s.roll_number}</div>
                    <div><b>Reg No:</b> {s.registration_number}</div>
                    <div><b>Subject:</b> {s.subject_name} ({s.subject_code})</div>
                    <div><b>Exam:</b> {s.exam_name}</div>
                    <div><b>Date:</b> {s.exam_date}</div>
                    <div><b>Sem / Branch:</b> Sem {s.semester} - {s.branch}</div>
                    <div><b>College:</b> {s.college_name}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Max Total Marks", rec.total_max_marks)
            m2.metric("AI Suggested Marks", rec.ai_total_marks)
            m3.metric("Final Awarded Marks", rec.final_total_marks)
            m4.metric("Percentage", f"{rec.percentage}%")
            m5.metric("Grade / Status", f"{grade_val} ({status_val})")

            st.markdown("---")
            st.write("##### Question-Wise Evaluation Review")

            updated_questions = []
            new_total_final = 0.0

            for idx, q in enumerate(rec.questions):
                with st.expander(f"📑 Question {q.question_id} | AI Suggested: {q.ai_marks}/{q.max_marks} | Teacher Final: {q.final_marks}/{q.max_marks}", expanded=True):
                    col_l, col_r = st.columns([1, 1])

                    with col_l:
                        st.markdown("**OCR Extracted Student Answer:**")
                        st.info(q.student_answer)

                        st.markdown("**Reference Model Answer:**")
                        st.caption(q.model_answer)

                    with col_r:
                        st.markdown(f"**Semantic Similarity:** `{int(q.similarity_score * 100)}%`")
                        st.progress(q.similarity_score)

                        st.write("**Matched Key Concepts:**")
                        m_html = "".join([f'<span class="pill-matched">✓ {k}</span>' for k in q.matched_keywords]) or "None"
                        st.markdown(m_html, unsafe_allow_html=True)

                        st.write("**Missing Key Concepts:**")
                        miss_html = "".join([f'<span class="pill-missing">✗ {k}</span>' for k in q.missing_keywords]) or "None"
                        st.markdown(miss_html, unsafe_allow_html=True)

                        st.markdown("**AI Explanation & Feedback:**")
                        st.warning(q.feedback)

                        co1, co2 = st.columns([1, 2])
                        with co1:
                            final_input = st.number_input(
                                f"Final Marks ({q.question_id})",
                                min_value=0.0,
                                max_value=float(q.max_marks),
                                value=float(q.final_marks),
                                step=0.5,
                                key=f"eval_marks_{q.question_id}_{idx}"
                            )
                        with co2:
                            comment_input = st.text_input(
                                "Teacher Comments",
                                value=q.teacher_comments or "",
                                key=f"eval_comm_{q.question_id}_{idx}"
                            )

                        q.final_marks = final_input
                        q.teacher_comments = comment_input
                        new_total_final += final_input
                        updated_questions.append(q)

            st.markdown("---")
            c_app1, c_app2 = st.columns([2, 1])
            with c_app1:
                st.write(f"Calculated Final Score: **{new_total_final} / {rec.total_max_marks}** ({round((new_total_final / rec.total_max_marks * 100), 2) if rec.total_max_marks > 0 else 0}%)")
            with c_app2:
                if st.button("✅ APPROVE FINAL RESULT", type="primary"):
                    rec.questions = updated_questions
                    rec.final_total_marks = round(new_total_final, 1)
                    rec.percentage = round((new_total_final / rec.total_max_marks * 100.0), 2) if rec.total_max_marks > 0 else 0.0
                    rec.grade, _ = determine_grade_and_status(rec.percentage, sys_settings)
                    rec.is_approved = True

                    db.update_evaluation(rec)
                    st.success("Final result approved and recorded in SQLite!")
                    st.rerun()

elif nav_choice == "📈 All Students Results":
    render_results_page(db)

elif nav_choice == "🔍 Student Search & Filters":
    render_student_details_page(db)

elif nav_choice == "📄 Reports & Exports":
    render_reports_page(db)

elif nav_choice == "⚙️ Settings":
    render_settings_page(db)
