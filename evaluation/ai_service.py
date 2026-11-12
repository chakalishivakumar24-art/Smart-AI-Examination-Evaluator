import os
from typing import Dict, List, Optional, Tuple
from evaluation.similarity import compute_similarity
from evaluation.grading import evaluate_keyword_coverage, calculate_completeness_score, calculate_question_marks

def evaluate_answer_pair(student_answer: str, model_answer: str, max_marks: float, keywords: List[str] = None) -> Tuple[float, float, List[str], List[str], str]:
    """
    Evaluates a single student answer against reference model answer.
    Returns (ai_marks, similarity_score, matched_keywords, missing_keywords, feedback).
    """
    if not student_answer or student_answer.strip() == "[Not Answered]":
        return 0.0, 0.0, [], keywords or [], "Question was left unanswered."

    sim_score = compute_similarity(student_answer, model_answer) if model_answer else 0.5
    matched_kw, missing_kw, kw_ratio = evaluate_keyword_coverage(student_answer, model_answer, keywords)
    comp_score = calculate_completeness_score(student_answer, model_answer) if model_answer else 1.0

    ai_marks = calculate_question_marks(sim_score, kw_ratio, comp_score, max_marks)
    
    pct = (ai_marks / max_marks) * 100 if max_marks > 0 else 0
    if pct >= 90:
        feedback = "Excellent answer! Fully matches key concepts and technical definitions."
    elif pct >= 75:
        feedback = "Good answer. Captures most required concepts."
    elif pct >= 50:
        feedback = "Partially correct. Demonstrates basic understanding but missing key details."
    else:
        feedback = "Incomplete or inaccurate answer. Major concepts missing."

    if missing_kw:
        feedback += f" Missing key concept(s): {', '.join(missing_kw[:3])}."

    return ai_marks, sim_score, matched_kw, missing_kw, feedback
