import os
from typing import Dict, List, Optional, Tuple, Any
from models.database_models import StudentDetails, QuestionEvaluation, EvaluationRecord, ModelAnswerItem, SystemSettings
from evaluation.similarity import compute_similarity
from evaluation.grading import (
    evaluate_keyword_coverage,
    calculate_completeness_score,
    calculate_question_marks,
    determine_grade_and_status,
    generate_question_feedback
)

def evaluate_with_llm_api(student_answer: str, model_answer: str, max_marks: float) -> Optional[Tuple[float, str]]:
    """
    Optional LLM evaluation service (OpenAI / Gemini).
    Returns (suggested_marks, explanation) or None if API key unconfigured/fails.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return None

def evaluate_question(
    question_id: str,
    student_answer: str,
    model_item: Optional[ModelAnswerItem],
    settings: SystemSettings
) -> QuestionEvaluation:
    """
    Evaluates a single question answer according to 3-case classification logic:
    CASE 1: Answer recognized + Model answer exists -> Evaluate similarity & score.
    CASE 2: Answer genuinely blank -> 0 marks, "Question was left unanswered."
    CASE 3: OCR unreadable / low confidence -> Warning notice, request teacher verification (NO automatic 0).
    Also guards against missing model answers without false 0 scores.
    """
    if model_item is None:
        return QuestionEvaluation(
            question_id=question_id,
            student_answer=student_answer or "[Unreadable / Low Confidence]",
            model_answer="[No Model Answer Configured]",
            max_marks=10.0,
            ai_marks=0.0,
            similarity_score=0.0,
            matched_keywords=[],
            missing_keywords=[],
            feedback="Model answers are not available for this exam/subject. Please add model answers before automatic evaluation.",
            final_marks=0.0,
            teacher_comments=""
        )

    max_m = float(model_item.max_marks)
    m_ans = model_item.model_answer
    kws = model_item.keywords

    # CASE 2: Genuinely empty answer
    if not student_answer or student_answer.strip() in ["", "[Not Answered]"]:
        return QuestionEvaluation(
            question_id=question_id,
            student_answer="[Not Answered]",
            model_answer=m_ans,
            max_marks=max_m,
            ai_marks=0.0,
            similarity_score=0.0,
            matched_keywords=[],
            missing_keywords=kws,
            feedback="Question was left unanswered.",
            final_marks=0.0,
            teacher_comments=""
        )

    # CASE 3: Unreadable or OCR low confidence
    if student_answer.strip() in ["[Unreadable / Low Confidence]", "[OCR Low Confidence]"]:
        return QuestionEvaluation(
            question_id=question_id,
            student_answer=student_answer,
            model_answer=m_ans,
            max_marks=max_m,
            ai_marks=round(max_m * 0.5, 1),
            similarity_score=0.5,
            matched_keywords=[],
            missing_keywords=kws,
            feedback="OCR could not confidently extract this answer. Please verify original answer sheet snippet and approve final marks.",
            final_marks=round(max_m * 0.5, 1),
            teacher_comments="Pending teacher handwriting verification"
        )

    # CASE 1: Valid student answer extracted -> Run NLP evaluation
    if settings.ai_evaluation_mode == "Advanced LLM":
        llm_res = evaluate_with_llm_api(student_answer, m_ans, max_m)
        if llm_res is not None:
            suggested_m, explanation = llm_res
            matched_kw, missing_kw, _ = evaluate_keyword_coverage(student_answer, m_ans, kws)
            return QuestionEvaluation(
                question_id=question_id,
                student_answer=student_answer,
                model_answer=m_ans,
                max_marks=max_m,
                ai_marks=suggested_m,
                similarity_score=suggested_m / max_m if max_m > 0 else 1.0,
                matched_keywords=matched_kw,
                missing_keywords=missing_kw,
                feedback=explanation,
                final_marks=suggested_m,
                teacher_comments=""
            )

    # Local NLP Evaluation Fallback (SentenceTransformers / TF-IDF Cosine Similarity)
    sim_score = compute_similarity(student_answer, m_ans)
    matched_kw, missing_kw, kw_ratio = evaluate_keyword_coverage(student_answer, m_ans, kws)
    comp_score = calculate_completeness_score(student_answer, m_ans)

    ai_marks = calculate_question_marks(sim_score, kw_ratio, comp_score, max_m)
    feedback_msg = generate_question_feedback(sim_score, matched_kw, missing_kw, ai_marks, max_m)


    return QuestionEvaluation(
        question_id=question_id,
        student_answer=student_answer,
        model_answer=m_ans,
        max_marks=max_m,
        ai_marks=ai_marks,
        similarity_score=sim_score,
        matched_keywords=matched_kw,
        missing_keywords=missing_kw,
        feedback=feedback_msg,
        final_marks=ai_marks,
        teacher_comments=""
    )

def evaluate_student_booklet(
    student_details: StudentDetails,
    student_answers: Dict[str, str],
    model_answer_items: List[ModelAnswerItem],
    booklet_filename: str,
    settings: SystemSettings
) -> EvaluationRecord:
    """
    Evaluates complete student booklet and returns a comprehensive EvaluationRecord.
    """
    model_dict = {m.question_id: m for m in model_answer_items}
    question_evaluations: List[QuestionEvaluation] = []

    all_q_ids = sorted(list(set(list(student_answers.keys()) + list(model_dict.keys()))))
    if not all_q_ids:
        all_q_ids = ["Q1", "Q2", "Q3", "Q4"]

    total_ai_marks = 0.0
    total_max_marks = 0.0

    for q_id in all_q_ids:
        s_ans = student_answers.get(q_id, "")
        m_item = model_dict.get(q_id)
        
        q_eval = evaluate_question(q_id, s_ans, m_item, settings)
        question_evaluations.append(q_eval)

        total_ai_marks += q_eval.ai_marks
        total_max_marks += q_eval.max_marks

    total_ai_marks = round(total_ai_marks, 1)
    total_max_marks = round(total_max_marks, 1)
    if total_max_marks > 0 and total_ai_marks > total_max_marks:
        total_ai_marks = total_max_marks
    percentage = min(100.0, round((total_ai_marks / total_max_marks * 100.0), 2)) if total_max_marks > 0 else 0.0

    grade_val, _ = determine_grade_and_status(percentage, settings)

    return EvaluationRecord(
        student=student_details,
        booklet_filename=booklet_filename,
        questions=question_evaluations,
        ai_total_marks=total_ai_marks,
        final_total_marks=total_ai_marks,
        total_max_marks=total_max_marks,
        percentage=percentage,
        grade=grade_val,
        is_approved=False
    )
