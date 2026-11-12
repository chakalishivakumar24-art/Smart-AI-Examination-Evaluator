from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

@dataclass
class User:
    id: Optional[int] = None
    username: str = ""
    password_hash: str = ""
    role: str = "Teacher"  # Admin or Teacher
    name: str = ""

@dataclass
class SystemSettings:
    pass_percentage: float = 40.0
    ai_evaluation_mode: str = "Local AI/NLP"  # "Local AI/NLP" or "Advanced LLM"
    ocr_engine: str = "paddleocr"
    grade_a_plus: float = 90.0
    grade_a: float = 80.0
    grade_b: float = 70.0
    grade_c: float = 60.0
    grade_d: float = 50.0
    grade_e: float = 40.0

@dataclass
class StudentDetails:
    student_name: str = ""
    roll_number: str = ""
    registration_number: str = ""
    subject_name: str = ""
    subject_code: str = ""
    exam_name: str = ""
    exam_date: str = ""
    college_name: str = ""
    semester: str = ""
    branch: str = ""
    department: str = ""
    section: str = ""
    class_year: str = ""
    booklet_serial_number: str = ""
    test_number: str = ""
    instructor_name: str = ""
    additional_sheets: str = ""
    examiner_remarks: str = ""
    recheck_request: str = ""
    student_signature_present: bool = True
    invigilator_signature_present: bool = True
    examiner_signature_present: bool = True
    course_title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        def sanitize(v: Any) -> str:
            if v is None or not str(v).strip():
                return "Not detected"
            cleaned = str(v).strip()
            if cleaned in ["N/A", "None", "()", "Sem", "Subject: ()", "null", "NOT_DETECTED"]:
                return "Not detected"
            return cleaned

        return {
            "student_name": sanitize(self.student_name),
            "enrollment_number": sanitize(self.roll_number or self.registration_number),
            "roll_number": sanitize(self.roll_number),
            "registration_number": sanitize(self.registration_number),
            "subject_name": sanitize(self.subject_name or self.course_title),
            "course_title": sanitize(self.course_title or self.subject_name),
            "subject_code": sanitize(self.subject_code),
            "course_code": sanitize(self.subject_code),
            "exam_name": sanitize(self.exam_name),
            "exam_date": sanitize(self.exam_date),
            "college_name": sanitize(self.college_name),
            "semester": sanitize(self.semester),
            "department": sanitize(self.department or self.branch),
            "branch": sanitize(self.branch or self.department),
            "section": sanitize(self.section),
            "class_year": sanitize(self.class_year),
            "booklet_serial_number": sanitize(self.booklet_serial_number),
            "test_number": sanitize(self.test_number),
            "instructor_name": sanitize(self.instructor_name),
            "additional_sheets": sanitize(self.additional_sheets),
            "examiner_remarks": sanitize(self.examiner_remarks),
            "recheck_request": sanitize(self.recheck_request),
            "student_signature_present": self.student_signature_present,
            "invigilator_signature_present": self.invigilator_signature_present,
            "examiner_signature_present": self.examiner_signature_present
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StudentDetails":
        return cls(
            student_name=data.get("student_name", ""),
            roll_number=data.get("roll_number", "") or data.get("enrollment_number", ""),
            registration_number=data.get("registration_number", "") or data.get("enrollment_number", ""),
            subject_name=data.get("subject_name", "") or data.get("course_title", ""),
            course_title=data.get("course_title", "") or data.get("subject_name", ""),
            subject_code=data.get("subject_code", "") or data.get("course_code", ""),
            exam_name=data.get("exam_name", ""),
            exam_date=data.get("exam_date", ""),
            college_name=data.get("college_name", ""),
            semester=data.get("semester", ""),
            branch=data.get("branch", "") or data.get("department", ""),
            department=data.get("department", "") or data.get("branch", ""),
            section=data.get("section", ""),
            class_year=data.get("class_year", ""),
            booklet_serial_number=data.get("booklet_serial_number", ""),
            test_number=data.get("test_number", ""),
            instructor_name=data.get("instructor_name", ""),
            additional_sheets=data.get("additional_sheets", ""),
            examiner_remarks=data.get("examiner_remarks", ""),
            recheck_request=data.get("recheck_request", "")
        )


@dataclass
class AnswerItem:
    id: Optional[int] = None
    booklet_id: Optional[int] = None
    question_number: str = ""
    answer_text: str = ""
    page_number: int = 1

@dataclass
class QuestionEvaluation:
    question_id: str
    student_answer: str
    model_answer: str
    max_marks: float
    ai_marks: float
    final_marks: float
    similarity_score: float = 0.0
    matched_keywords: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    feedback: str = ""
    teacher_comments: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "student_answer": self.student_answer,
            "model_answer": self.model_answer,
            "max_marks": self.max_marks,
            "ai_marks": self.ai_marks,
            "final_marks": self.final_marks,
            "similarity_score": self.similarity_score,
            "matched_keywords": self.matched_keywords,
            "missing_keywords": self.missing_keywords,
            "feedback": self.feedback,
            "teacher_comments": self.teacher_comments
        }

@dataclass
class ModelAnswerItem:
    id: Optional[int] = None
    question_id: str = "Q1"
    question_text: str = ""
    model_answer: str = ""
    max_marks: float = 10.0
    keywords: List[str] = field(default_factory=list)
    subject_code: str = "CS501"
    subject_name: str = "Machine Learning"

@dataclass
class EvaluationRecord:
    id: Optional[int] = None
    booklet_filename: str = ""
    student: StudentDetails = field(default_factory=StudentDetails)
    total_max_marks: float = 15.0
    ai_total_marks: float = 0.0
    final_total_marks: float = 0.0
    percentage: float = 0.0
    grade: str = "F"
    is_approved: bool = False
    evaluation_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    questions: List[QuestionEvaluation] = field(default_factory=list)
