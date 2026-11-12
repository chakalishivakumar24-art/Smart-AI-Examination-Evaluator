import os
from io import BytesIO
from typing import Union
from models.database_models import EvaluationRecord, SystemSettings
from evaluation.grading import determine_grade_and_status
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(record: EvaluationRecord, settings: SystemSettings = None, output_path_or_buffer: Union[str, BytesIO] = None) -> Union[str, bytes]:
    """
    Generates a formal PDF evaluation report for a student using ReportLab.
    Returns path string if saved to file, or bytes if buffer provided.
    """
    if settings is None:
        settings = SystemSettings()

    grade, status = determine_grade_and_status(record.percentage, settings)
    is_buffer = isinstance(output_path_or_buffer, BytesIO) or output_path_or_buffer is None
    buffer = BytesIO() if is_buffer else output_path_or_buffer

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"),
        alignment=1
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4B5563"),
        alignment=1
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1F2937")
    )

    bold_body_style = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    story = []

    # Title Header
    college_title = record.student.college_name.upper() if record.student.college_name else "EXAMINATION EVALUATION BOARD"
    story.append(Paragraph(college_title, title_style))
    story.append(Paragraph("OFFICIAL STUDENT EXAMINATION EVALUATION REPORT", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1E3A8A"), spaceAfter=15))

    # Student Metadata Table
    student_data = [
        [
            Paragraph("<b>Student Name:</b>", body_style), Paragraph(record.student.student_name, bold_body_style),
            Paragraph("<b>Roll Number:</b>", body_style), Paragraph(record.student.roll_number, bold_body_style)
        ],
        [
            Paragraph("<b>Registration No:</b>", body_style), Paragraph(record.student.registration_number, body_style),
            Paragraph("<b>Subject:</b>", body_style), Paragraph(f"{record.student.subject_name} ({record.student.subject_code})", body_style)
        ],
        [
            Paragraph("<b>Exam Name:</b>", body_style), Paragraph(record.student.exam_name, body_style),
            Paragraph("<b>Exam Date:</b>", body_style), Paragraph(record.student.exam_date, body_style)
        ],
        [
            Paragraph("<b>Semester / Branch:</b>", body_style), Paragraph(f"Sem {record.student.semester} - {record.student.branch}", body_style),
            Paragraph("<b>Evaluation Date:</b>", body_style), Paragraph(record.evaluation_date[:10], body_style)
        ]
    ]

    info_table = Table(student_data, colWidths=[110, 160, 110, 160])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 15))

    # Result Summary Table
    status_color = "#16A34A" if status == "PASS" else "#DC2626"
    
    summary_data = [
        [
            Paragraph("<b>Total Maximum Marks</b>", body_style),
            Paragraph("<b>AI Suggested Marks</b>", body_style),
            Paragraph("<b>Final Marks Awarded</b>", body_style),
            Paragraph("<b>Percentage</b>", body_style),
            Paragraph("<b>Grade</b>", body_style),
            Paragraph("<b>Status</b>", body_style)
        ],
        [
            Paragraph(f"<font size=12><b>{record.total_max_marks}</b></font>", body_style),
            Paragraph(f"<font size=12><b>{record.ai_total_marks}</b></font>", body_style),
            Paragraph(f"<font size=12 color='#1E3A8A'><b>{record.final_total_marks}</b></font>", body_style),
            Paragraph(f"<font size=12 color='{status_color}'><b>{record.percentage}%</b></font>", body_style),
            Paragraph(f"<font size=14 color='{status_color}'><b>{grade}</b></font>", body_style),
            Paragraph(f"<font size=12 color='{status_color}'><b>{status}</b></font>", body_style)
        ]
    ]
    summary_table = Table(summary_data, colWidths=[90, 90, 90, 90, 90, 90])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E0F2FE")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#F0F9FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#0284C7")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))

    # Question-wise Summary Table
    story.append(Paragraph("Question-Wise Evaluation Breakdown", section_heading))
    
    q_table_data = [
        [
            Paragraph("<b>Question</b>", bold_body_style),
            Paragraph("<b>Max Marks</b>", bold_body_style),
            Paragraph("<b>AI Marks</b>", bold_body_style),
            Paragraph("<b>Final Marks</b>", bold_body_style),
            Paragraph("<b>Similarity</b>", bold_body_style),
            Paragraph("<b>Feedback & Explanation</b>", bold_body_style)
        ]
    ]

    for q in record.questions:
        sim_pct = f"{int(q.similarity_score * 100)}%"
        q_table_data.append([
            Paragraph(q.question_id, bold_body_style),
            Paragraph(str(q.max_marks), body_style),
            Paragraph(str(q.ai_marks), body_style),
            Paragraph(f"<b>{q.final_marks}</b>", bold_body_style),
            Paragraph(sim_pct, body_style),
            Paragraph(q.feedback, body_style)
        ])

    q_table = Table(q_table_data, colWidths=[60, 65, 60, 65, 60, 230])
    q_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#94A3B8")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(q_table)
    story.append(Spacer(1, 20))

    # Detailed Question Audit Section
    story.append(Paragraph("Detailed Answer Audit & Teacher Comments", section_heading))
    for q in record.questions:
        matched_str = ", ".join(q.matched_keywords) if q.matched_keywords else "None"
        missing_str = ", ".join(q.missing_keywords) if q.missing_keywords else "None"
        
        detail_text = f"""
        <b>{q.question_id} (Max Marks: {q.max_marks} | Final Marks: {q.final_marks})</b><br/>
        <b>Student Answer:</b> {q.student_answer}<br/>
        <b>Model Answer:</b> {q.model_answer}<br/>
        <font color="#15803D"><b>Matched Keywords:</b> {matched_str}</font><br/>
        <font color="#B91C1C"><b>Missing Keywords:</b> {missing_str}</font><br/>
        """
        if q.teacher_comments:
            detail_text += f"<b>Teacher Comments:</b> {q.teacher_comments}<br/>"

        q_box_data = [[Paragraph(detail_text, body_style)]]
        q_box_table = Table(q_box_data, colWidths=[540])
        q_box_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(KeepTogether([q_box_table, Spacer(1, 8)]))

    # Footer Signatures & Approval Status
    story.append(Spacer(1, 25))
    sig_data = [
        [
            Paragraph("<b>AI Evaluation Engine</b><br/>Status: Processed & Verified", body_style),
            Paragraph(f"<b>Teacher / Evaluator Sign-off</b><br/>Approval Status: <b>{'APPROVED & OFFICIAL' if record.is_approved else 'PENDING TEACHER APPROVAL'}</b>", body_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[270, 270])
    sig_table.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,-1), 1, colors.HexColor("#64748B")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(KeepTogether(sig_table))

    doc.build(story)

    if is_buffer:
        return buffer.getvalue()
    return output_path_or_buffer
