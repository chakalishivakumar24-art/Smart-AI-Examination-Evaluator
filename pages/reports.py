import streamlit as st
from database.database import EvaluationDatabase
from reports.report_generator import generate_pdf_report
from reports.excel_csv_exporter import export_to_csv, export_to_excel

def render_reports_page(db: EvaluationDatabase):
    st.subheader("Generate PDF Marksheets & Export Datasets (Excel / CSV)")
    sys_settings = db.get_settings()
    evals = db.search_evaluations()

    if evals:
        col_pdf, col_exp = st.columns([1, 1])

        with col_pdf:
            st.write("##### Individual PDF Marksheet Generator")
            selected_st = st.selectbox("Select Student for PDF Generation", [f"#{r.id} - {r.student.student_name} ({r.student.roll_number})" for r in evals])
            st_id = int(selected_st.split(" - ")[0].replace("#", ""))
            target_rec = db.get_evaluation(st_id)

            if target_rec:
                pdf_bytes = generate_pdf_report(target_rec, sys_settings)
                st.download_button(
                    label=f"📥 DOWNLOAD PDF ({target_rec.student.roll_number})",
                    data=pdf_bytes,
                    file_name=f"Evaluation_Report_{target_rec.student.roll_number}.pdf",
                    mime="application/pdf",
                    type="primary"
                )

        with col_exp:
            st.write("##### Bulk Data Exporters (Excel & CSV)")
            
            excel_bytes = export_to_excel(evals, sys_settings)
            st.download_button(
                label="📊 Export to Excel (.xlsx)",
                data=excel_bytes,
                file_name="Student_Evaluation_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            csv_bytes = export_to_csv(evals, sys_settings)
            st.download_button(
                label="📄 Export to CSV",
                data=csv_bytes,
                file_name="Student_Evaluation_Results.csv",
                mime="text/csv"
            )
    else:
        st.info("No evaluations available to export.")
