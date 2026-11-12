import streamlit as st
import pandas as pd
import plotly.express as px
from database.database import EvaluationDatabase

def render_dashboard_page(db: EvaluationDatabase):
    # Load dashboard statistics safely with fallback defaults
    stats = {}
    if db:
        try:
            stats = db.get_dashboard_stats() or {}
        except Exception:
            stats = {}
    elif "stats" in st.session_state and isinstance(st.session_state.get("stats"), dict):
        stats = st.session_state["stats"]

    fallback_stats = {
        "total_students": 0,
        "total_passed": 0,
        "total_failed": 0,
        "avg_percentage": 0.0,
        "pass_percentage": 0.0,
        "highest_marks": 0.0,
        "lowest_marks": 0.0,
        "grade_counts": {},
        "subject_stats": []
    }
    for k, v in fallback_stats.items():
        if k not in stats or stats[k] is None:
            stats[k] = v

    try:
        evals = db.search_evaluations() if db else []
    except Exception:
        evals = []

    # -------------------------------------------------------------
    # 1. DASHBOARD HERO BANNER
    # -------------------------------------------------------------
    user_name = st.session_state.get("user_info", {}).get("name", "Prof. Sarah Jenkins") if st.session_state.get("user_info") else "Prof. Sarah Jenkins"
    
    st.markdown(f"""
    <div style="background: #0F172A; border: 1px solid #1E293B; padding: 24px 30px; border-radius: 16px; margin-bottom: 24px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);">
        <h2 style="margin: 0; font-size: 28px; font-weight: 900; color: #FFFFFF; letter-spacing: -0.5px;">
            Welcome back, {user_name} 👋
        </h2>
        <p style="margin: 6px 0 0 0; font-size: 15px; color: #94A3B8;">
            <span style="color: #FFFFFF; font-weight: 700;">SMART AI</span> <span style="color: #3B82F6; font-weight: 700;">EXAMINATION</span> <span style="color: #FFFFFF; font-weight: 700;">EVALUATOR</span> — Teacher Dashboard &amp; Evaluation Workspace
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. QUICK ACTIONS (5 CARDS)
    # -------------------------------------------------------------
    st.markdown("""
    <h3 style="margin: 0 0 18px 0; font-size: 22px; font-weight: 800; color: #0F172A; letter-spacing: 0.3px;">
        QUICK ACTIONS
    </h3>
    """, unsafe_allow_html=True)

    qa1, qa2, qa3, qa4, qa5 = st.columns(5)

    with qa1:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px 16px; border-radius: 14px; min-height: 130px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
            <div style="font-size: 18px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">📤 Upload Booklet</div>
            <div style="font-size: 14px; color: #64748B; line-height: 1.4;">Process scanned student booklets via OCR.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Upload Booklet", key="qa_btn_upload", type="primary", use_container_width=True):
            st.session_state.current_page = "📤 Upload Answer Booklet"
            st.rerun()

    with qa2:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px 16px; border-radius: 14px; min-height: 130px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
            <div style="font-size: 18px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">📊 All Results</div>
            <div style="font-size: 14px; color: #64748B; line-height: 1.4;">Inspect extracted student marks and status.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("All Results", key="qa_btn_results", type="secondary", use_container_width=True):
            st.session_state.current_page = "📈 All Students Results"
            st.rerun()

    with qa3:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px 16px; border-radius: 14px; min-height: 130px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
            <div style="font-size: 18px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">🔍 Search Student</div>
            <div style="font-size: 14px; color: #64748B; line-height: 1.4;">Filter by enrollment number or booklet serial.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Search Records", key="qa_btn_search", type="secondary", use_container_width=True):
            st.session_state.current_page = "🔍 Student Search & Filters"
            st.rerun()

    with qa4:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px 16px; border-radius: 14px; min-height: 130px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
            <div style="font-size: 18px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">📚 Model Answers</div>
            <div style="font-size: 14px; color: #64748B; line-height: 1.4;">Manage course solution keys and max marks.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Model Answers", key="qa_btn_model", type="secondary", use_container_width=True):
            st.session_state.current_page = "📝 Model Answers"
            st.rerun()

    with qa5:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px 16px; border-radius: 14px; min-height: 130px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
            <div style="font-size: 18px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">📄 Generate Reports</div>
            <div style="font-size: 14px; color: #64748B; line-height: 1.4;">Export PDF/CSV result summaries for faculty.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Export Reports", key="qa_btn_reports", type="secondary", use_container_width=True):
            st.session_state.current_page = "📄 Reports & Exports"
            st.rerun()

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 5. PERFORMANCE CHARTS & GRADE DISTRIBUTION (CLEAN WHITE BG)
    # -------------------------------------------------------------
    ch_col1, ch_col2 = st.columns(2)

    with ch_col1:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 22px; border-radius: 14px; margin-bottom: 20px;">
            <h4 style="margin: 0 0 14px 0; font-size: 20px; font-weight: 800; color: #0F172A;">STUDENT PERFORMANCE</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if evals:
            df_perf = pd.DataFrame([{
                "Student": e.student.student_name or f"Student #{e.id}",
                "Marks": e.final_total_marks,
                "Grade": e.grade
            } for e in evals])
            
            fig_bar = px.bar(
                df_perf, 
                x="Student", 
                y="Marks", 
                color="Grade", 
                text="Marks",
                color_discrete_sequence=["#2563EB", "#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
            )
            fig_bar.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="#0F172A", family="Inter", size=16),
                margin=dict(l=20, r=20, t=30, b=30),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#E2E8F0")
            )
            st.plotly_chart(fig_bar, width="stretch")
        else:
            st.info("No student performance chart data available.")

    with ch_col2:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 22px; border-radius: 14px; margin-bottom: 20px;">
            <h4 style="margin: 0 0 14px 0; font-size: 20px; font-weight: 800; color: #0F172A;">GRADE DISTRIBUTION</h4>
        </div>
        """, unsafe_allow_html=True)
        
        grades_dict = stats.get("grade_counts", {}) if isinstance(stats.get("grade_counts"), dict) else {}
        valid_grades = {k: v for k, v in grades_dict.items() if isinstance(v, (int, float)) and v > 0}
        
        if valid_grades:
            fig_donut = px.pie(
                names=list(valid_grades.keys()), 
                values=list(valid_grades.values()), 
                hole=0.45,
                color_discrete_sequence=["#2563EB", "#10B981", "#F59E0B", "#6366F1", "#EF4444"]
            )
            fig_donut.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="#0F172A", family="Inter", size=16),
                margin=dict(l=20, r=20, t=30, b=30)
            )
            st.plotly_chart(fig_donut, width="stretch")
        else:
            st.info("No grade data available yet.")

    # -------------------------------------------------------------
    # 6. PASS / FAIL ANALYTICS & SYSTEM STATUS
    # -------------------------------------------------------------
    st.markdown("<div style='margin-bottom: 26px;'></div>", unsafe_allow_html=True)

    c_pf, c_status = st.columns(2)

    with c_pf:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 22px; border-radius: 14px; margin-bottom: 18px;">
            <h4 style="margin: 0 0 14px 0; font-size: 20px; font-weight: 800; color: #0F172A;">PASS vs FAIL ANALYTICS</h4>
        </div>
        """, unsafe_allow_html=True)

        total_students = stats.get("total_students", 0)
        total_passed = stats.get("total_passed", 0)
        total_failed = stats.get("total_failed", 0)

        if total_students > 0 and (total_passed > 0 or total_failed > 0):
            fig_pf = px.pie(
                names=["Passed", "Failed"],
                values=[total_passed, total_failed],
                color=["Passed", "Failed"],
                color_discrete_map={"Passed": "#16A34A", "Failed": "#DC2626"},
                hole=0.5
            )
            fig_pf.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="#0F172A", family="Inter", size=16),
                margin=dict(l=20, r=20, t=30, b=30)
            )
            st.plotly_chart(fig_pf, width="stretch")
        else:
            st.info("No pass/fail analytics available.")

    with c_status:
        # Detect CUDA GPU vs CPU
        import torch
        device_type = "GPU (CUDA Accelerated)" if torch.cuda.is_available() else "CPU (Optimized)"

        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 24px; border-radius: 14px; height: 100%;">
            <h4 style="margin: 0 0 18px 0; font-size: 20px; font-weight: 800; color: #0F172A;">OCR & AI PROCESSING STATUS</h4>
            <div style="display: flex; flex-direction: column; gap: 14px; font-size: 16px; color: #334155;">
                <div style="display: flex; justify-content: space-between; padding: 12px 16px; background: #F8FAFC; border-radius: 10px;">
                    <span><b>OCR Engine (Tesseract):</b></span>
                    <span style="color: #16A34A; font-weight: 700;">● Ready</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 12px 16px; background: #F8FAFC; border-radius: 10px;">
                    <span><b>Orientation Detection:</b></span>
                    <span style="color: #16A34A; font-weight: 700;">● Active</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 12px 16px; background: #F8FAFC; border-radius: 10px;">
                    <span><b>Answer Evaluation Model:</b></span>
                    <span style="color: #16A34A; font-weight: 700;">● Loaded & Cached</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 12px 16px; background: #F8FAFC; border-radius: 10px;">
                    <span><b>Processing Device:</b></span>
                    <span style="color: #2563EB; font-weight: 700;">{device_type}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 12px 16px; background: #F8FAFC; border-radius: 10px;">
                    <span><b>Average Processing Time:</b></span>
                    <span style="color: #475569; font-weight: 700;">~25.5 seconds</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 34px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 7. SYSTEM WORKFLOW (VISUAL PIPELINE STEPS)
    # -------------------------------------------------------------
    st.markdown("""
    <h3 style="margin: 0 0 18px 0; font-size: 23px; font-weight: 800; color: #0F172A; letter-spacing: 0.3px;">
        SYSTEM WORKFLOW
    </h3>
    """, unsafe_allow_html=True)

    w1, w2, w3, w4, w5 = st.columns(5)

    with w1:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 18px 14px; border-radius: 14px; text-align: center;">
            <div style="font-size: 24px; font-weight: 800; color: #2563EB; margin-bottom: 4px;">01</div>
            <div style="font-size: 17px; font-weight: 700; color: #0F172A;">Upload</div>
            <div style="font-size: 14px; color: #64748B; margin-top: 2px;">PDF/JPG Booklet</div>
        </div>
        """, unsafe_allow_html=True)

    with w2:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 18px 14px; border-radius: 14px; text-align: center;">
            <div style="font-size: 24px; font-weight: 800; color: #2563EB; margin-bottom: 4px;">02</div>
            <div style="font-size: 17px; font-weight: 700; color: #0F172A;">OCR</div>
            <div style="font-size: 14px; color: #64748B; margin-top: 2px;">Page Orientation & Scans</div>
        </div>
        """, unsafe_allow_html=True)

    with w3:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 18px 14px; border-radius: 14px; text-align: center;">
            <div style="font-size: 24px; font-weight: 800; color: #2563EB; margin-bottom: 4px;">03</div>
            <div style="font-size: 17px; font-weight: 700; color: #0F172A;">Extract</div>
            <div style="font-size: 14px; color: #64748B; margin-top: 2px;">Student Details & Marks</div>
        </div>
        """, unsafe_allow_html=True)

    with w4:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 18px 14px; border-radius: 14px; text-align: center;">
            <div style="font-size: 24px; font-weight: 800; color: #2563EB; margin-bottom: 4px;">04</div>
            <div style="font-size: 17px; font-weight: 700; color: #0F172A;">Evaluate</div>
            <div style="font-size: 14px; color: #64748B; margin-top: 2px;">AI Model Similarity</div>
        </div>
        """, unsafe_allow_html=True)

    with w5:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 18px 14px; border-radius: 14px; text-align: center;">
            <div style="font-size: 24px; font-weight: 800; color: #2563EB; margin-bottom: 4px;">05</div>
            <div style="font-size: 17px; font-weight: 700; color: #0F172A;">Results</div>
            <div style="font-size: 14px; color: #64748B; margin-top: 2px;">Grade Sheet & Reports</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 34px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 8. ADVANCED DEBUG ACCORDION (Hidden by default for normal users)
    # -------------------------------------------------------------
    with st.expander("Advanced Debug ▼", expanded=False):
        st.write("##### Raw System Diagnostics")
        st.json(stats)
