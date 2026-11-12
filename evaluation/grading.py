from typing import List, Tuple, Dict
from utils.text_parser import extract_keywords, clean_text
from models.database_models import SystemSettings

def evaluate_keyword_coverage(student_answer: str, model_answer: str, target_keywords: List[str] = None) -> Tuple[List[str], List[str], float]:
    """
    Checks keyword match between student answer and reference model answer keywords.
    Returns (matched_keywords, missing_keywords, coverage_ratio).
    """
    if not target_keywords:
        target_keywords = extract_keywords(model_answer, top_n=10)

    cleaned_student = clean_text(student_answer)
    matched = []
    missing = []

    for kw in target_keywords:
        kw_clean = clean_text(kw)
        if kw_clean in cleaned_student or any(word in cleaned_student for word in kw_clean.split()):
            matched.append(kw)
        else:
            missing.append(kw)

    coverage_ratio = len(matched) / len(target_keywords) if target_keywords else 1.0
    return matched, missing, round(coverage_ratio, 4)

def calculate_completeness_score(student_answer: str, model_answer: str) -> float:
    """Estimates answer length and depth completeness relative to model answer."""
    len_s = len(clean_text(student_answer).split())
    len_m = len(clean_text(model_answer).split())

    if len_m == 0:
        return 1.0
    ratio = len_s / float(len_m)
    if ratio >= 0.8:
        return 1.0
    elif ratio >= 0.5:
        return 0.85
    elif ratio >= 0.2:
        return 0.6
    else:
        return 0.3

def calculate_question_marks(
    similarity_score: float,
    keyword_ratio: float,
    completeness_score: float,
    max_marks: float
) -> float:
    """
    Calculates AI suggested marks based on weighted criteria:
    - 50% Semantic Similarity
    - 35% Keyword Coverage
    - 15% Completeness
    Returns assigned mark bounded between 0 and max_marks (rounded to 1 decimal place).
    """
    if similarity_score < 0.15:
        return 0.0

    raw_percentage = (0.50 * similarity_score) + (0.35 * keyword_ratio) + (0.15 * completeness_score)
    assigned_marks = round(raw_percentage * max_marks, 1)
    
    assigned_marks = max(0.0, min(max_marks, assigned_marks))
    return assigned_marks

def calculate_grade(percentage: float, settings: SystemSettings = None) -> str:
    """
    Calculates letter grade from overall percentage:
    >= 90 = A+
    >= 80 = A
    >= 70 = B
    >= 60 = C
    >= 50 = D
    >= 40 = E
    Below 40 = F
    """
    pct = float(percentage) if percentage is not None else 0.0
    if settings:
        if pct >= float(getattr(settings, "grade_a_plus", 90.0)):
            return "A+"
        elif pct >= float(getattr(settings, "grade_a", 80.0)):
            return "A"
        elif pct >= float(getattr(settings, "grade_b", 70.0)):
            return "B"
        elif pct >= float(getattr(settings, "grade_c", 60.0)):
            return "C"
        elif pct >= float(getattr(settings, "grade_d", 50.0)):
            return "D"
        elif pct >= float(getattr(settings, "grade_e", 40.0)):
            return "E"
        else:
            return "F"

    if pct >= 90.0:
        return "A+"
    elif pct >= 80.0:
        return "A"
    elif pct >= 70.0:
        return "B"
    elif pct >= 60.0:
        return "C"
    elif pct >= 50.0:
        return "D"
    elif pct >= 40.0:
        return "E"
    else:
        return "F"


def determine_grade_and_status(percentage: float, settings: SystemSettings = None) -> Tuple[str, str]:
    """
    Determines letter grade and pass/fail status from overall percentage.
    PASS if percentage >= pass_percentage threshold (default 40%), FAIL otherwise.
    """
    if settings is None:
        settings = SystemSettings()

    grade = calculate_grade(percentage, settings)
    pass_thresh = settings.pass_percentage
    status = "PASS" if percentage >= pass_thresh else "FAIL"
    return grade, status


def generate_question_feedback(
    similarity_score: float,
    matched: List[str],
    missing: List[str],
    assigned_marks: float,
    max_marks: float
) -> str:
    """Generates constructive AI explanation and feedback."""
    pct = (assigned_marks / max_marks) * 100 if max_marks > 0 else 0
    if pct >= 90:
        feedback = "Excellent answer! Fully matches key concepts and technical definitions."
    elif pct >= 75:
        feedback = "Good answer. Captures most required concepts."
    elif pct >= 50:
        feedback = "Partially correct. Demonstrates basic understanding but missing some key concepts."
    else:
        feedback = "Incomplete or inaccurate answer. Major concepts missing."

    if missing:
        feedback += f" Missing key concept(s): {', '.join(missing[:3])}."
    return feedback
