import sqlite3
import json
import os
from typing import List, Dict, Optional, Any
from models.database_models import StudentDetails, QuestionEvaluation, EvaluationRecord, ModelAnswerItem, AnswerItem, SystemSettings

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "database.db")

class EvaluationDatabase:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # System Settings Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    pass_percentage REAL DEFAULT 40.0,
                    ai_evaluation_mode TEXT DEFAULT 'Local AI/NLP',
                    ocr_engine TEXT DEFAULT 'yolo_easyocr',
                    grade_a_plus REAL DEFAULT 90.0,
                    grade_a REAL DEFAULT 80.0,
                    grade_b REAL DEFAULT 70.0,
                    grade_c REAL DEFAULT 60.0,
                    grade_d REAL DEFAULT 50.0,
                    grade_e REAL DEFAULT 40.0
                )
            """)

            # Seed default settings if empty
            cursor.execute("SELECT COUNT(*) FROM settings")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO settings (id) VALUES (1)")

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    role TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Students table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_name TEXT,
                    roll_number TEXT,
                    registration_number TEXT,
                    college_name TEXT,
                    semester TEXT,
                    branch TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Exams table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_name TEXT,
                    subject_code TEXT,
                    exam_name TEXT,
                    exam_date TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Booklets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS booklets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    exam_id INTEGER,
                    filename TEXT,
                    page_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students(id),
                    FOREIGN KEY (exam_id) REFERENCES exams(id)
                )
            """)

            # Answers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    booklet_id INTEGER,
                    question_number TEXT,
                    answer_text TEXT,
                    page_number INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (booklet_id) REFERENCES booklets(id)
                )
            """)

            # Master Evaluations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    booklet_filename TEXT,
                    student_name TEXT,
                    roll_number TEXT,
                    registration_number TEXT,
                    subject_name TEXT,
                    subject_code TEXT,
                    exam_name TEXT,
                    exam_date TEXT,
                    college_name TEXT,
                    semester TEXT,
                    branch TEXT,
                    total_max_marks REAL,
                    ai_total_marks REAL,
                    final_total_marks REAL,
                    percentage REAL,
                    grade TEXT,
                    is_approved INTEGER DEFAULT 0,
                    evaluation_date TEXT
                )
            """)

            # Question evaluations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS question_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id INTEGER,
                    question_id TEXT,
                    student_answer TEXT,
                    model_answer TEXT,
                    max_marks REAL,
                    ai_marks REAL,
                    final_marks REAL,
                    similarity_score REAL,
                    matched_keywords TEXT,
                    missing_keywords TEXT,
                    feedback TEXT,
                    teacher_comments TEXT,
                    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id) ON DELETE CASCADE
                )
            """)

            # Model answers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_code TEXT,
                    subject_name TEXT,
                    question_id TEXT,
                    question_text TEXT,
                    model_answer TEXT,
                    max_marks REAL,
                    keywords TEXT
                )
            """)

            # Field verifications table for persistent dashboard verification statuses & edited values
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_verifications (
                    student_uid TEXT PRIMARY KEY,
                    booklet_filename TEXT,
                    roll_number TEXT,
                    field_data TEXT,
                    accuracy_pct REAL DEFAULT 0.0,
                    error_pct REAL DEFAULT 0.0,
                    pending_pct REAL DEFAULT 100.0,
                    verified_count INTEGER DEFAULT 0,
                    incorrect_count INTEGER DEFAULT 0,
                    pending_count INTEGER DEFAULT 11,
                    evaluation_id INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    # Settings CRUD
    def get_settings(self) -> SystemSettings:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM settings WHERE id=1")
            row = cursor.fetchone()
            if not row:
                return SystemSettings()
            raw_ocr = str(row["ocr_engine"]).strip() if row["ocr_engine"] else "paddleocr"
            valid_keys = ["paddleocr", "yolo_paddleocr", "yolo_easyocr", "yolo_ocr", "hybrid", "easyocr", "tesseract"]
            if raw_ocr not in valid_keys:
                raw_ocr = "paddleocr"

            return SystemSettings(
                pass_percentage=row["pass_percentage"],
                ai_evaluation_mode=row["ai_evaluation_mode"],
                ocr_engine=raw_ocr,
                grade_a_plus=row["grade_a_plus"],
                grade_a=row["grade_a"],
                grade_b=row["grade_b"],
                grade_c=row["grade_c"],
                grade_d=row["grade_d"],
                grade_e=row["grade_e"]
            )

    def update_settings(self, settings: SystemSettings):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE settings SET
                    pass_percentage=?, ai_evaluation_mode=?, ocr_engine=?,
                    grade_a_plus=?, grade_a=?, grade_b=?, grade_c=?, grade_d=?, grade_e=?
                WHERE id=1
            """, (
                settings.pass_percentage, settings.ai_evaluation_mode, settings.ocr_engine,
                settings.grade_a_plus, settings.grade_a, settings.grade_b, settings.grade_c,
                settings.grade_d, settings.grade_e
            ))
            conn.commit()

    # Model Answers Full CRUD
    def add_model_answer(self, item: ModelAnswerItem):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_answers (subject_code, subject_name, question_id, question_text, model_answer, max_marks, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                item.subject_code, item.subject_name, item.question_id, item.question_text,
                item.model_answer, item.max_marks, json.dumps(item.keywords)
            ))
            conn.commit()

    def delete_model_answer(self, model_id: int):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM model_answers WHERE id=?", (model_id,))
            conn.commit()

    def save_model_answers(self, subject_code: str, subject_name: str, items: List[ModelAnswerItem]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM model_answers WHERE subject_code=?", (subject_code,))
            for item in items:
                cursor.execute("""
                    INSERT INTO model_answers (
                        subject_code, subject_name, question_id, question_text, model_answer, max_marks, keywords
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    subject_code, subject_name, item.question_id, item.question_text,
                    item.model_answer, item.max_marks, json.dumps(item.keywords)
                ))
            conn.commit()

    def get_model_answers(self, subject_code: str) -> List[ModelAnswerItem]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_answers WHERE subject_code=? ORDER BY question_id ASC", (subject_code,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append(ModelAnswerItem(
                    id=r["id"],
                    subject_code=r["subject_code"],
                    subject_name=r["subject_name"],
                    question_id=r["question_id"],
                    question_text=r["question_text"],
                    model_answer=r["model_answer"],
                    max_marks=r["max_marks"],
                    keywords=json.loads(r["keywords"] or "[]")
                ))
            return results

    def create_booklet_record(self, student: StudentDetails, filename: str, page_count: int, extracted_answers: Dict[str, str]) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO students (student_name, roll_number, registration_number, college_name, semester, branch)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (student.student_name, student.roll_number, student.registration_number, student.college_name, student.semester, student.branch))
            student_id = cursor.lastrowid

            cursor.execute("""
                INSERT INTO exams (subject_name, subject_code, exam_name, exam_date)
                VALUES (?, ?, ?, ?)
            """, (student.subject_name, student.subject_code, student.exam_name, student.exam_date))
            exam_id = cursor.lastrowid

            cursor.execute("""
                INSERT INTO booklets (student_id, exam_id, filename, page_count)
                VALUES (?, ?, ?, ?)
            """, (student_id, exam_id, filename, page_count))
            booklet_id = cursor.lastrowid

            for q_num, ans_text in extracted_answers.items():
                cursor.execute("""
                    INSERT INTO answers (booklet_id, question_number, answer_text, page_number)
                    VALUES (?, ?, ?, ?)
                """, (booklet_id, q_num, ans_text, 1))

            conn.commit()
            return booklet_id

    def save_evaluation(self, record: EvaluationRecord) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO evaluations (
                    booklet_filename, student_name, roll_number, registration_number,
                    subject_name, subject_code, exam_name, exam_date, college_name,
                    semester, branch, total_max_marks, ai_total_marks, final_total_marks,
                    percentage, grade, is_approved, evaluation_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.booklet_filename, record.student.student_name, record.student.roll_number,
                record.student.registration_number, record.student.subject_name, record.student.subject_code,
                record.student.exam_name, record.student.exam_date, record.student.college_name,
                record.student.semester, record.student.branch, record.total_max_marks,
                record.ai_total_marks, record.final_total_marks, record.percentage,
                record.grade, 1 if record.is_approved else 0, record.evaluation_date
            ))

            eval_id = cursor.lastrowid

            for q in record.questions:
                cursor.execute("""
                    INSERT INTO question_evaluations (
                        evaluation_id, question_id, student_answer, model_answer,
                        max_marks, ai_marks, final_marks, similarity_score,
                        matched_keywords, missing_keywords, feedback, teacher_comments
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    eval_id, q.question_id, q.student_answer, q.model_answer,
                    q.max_marks, q.ai_marks, q.final_marks, q.similarity_score,
                    json.dumps(q.matched_keywords), json.dumps(q.missing_keywords),
                    q.feedback, q.teacher_comments
                ))

            conn.commit()
            return eval_id

    def update_evaluation(self, record: EvaluationRecord) -> bool:
        if not record.id:
            return False
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE evaluations SET
                    student_name=?, roll_number=?, registration_number=?,
                    subject_name=?, subject_code=?, exam_name=?, exam_date=?,
                    college_name=?, semester=?, branch=?, total_max_marks=?,
                    ai_total_marks=?, final_total_marks=?, percentage=?, grade=?,
                    is_approved=?
                WHERE id=?
            """, (
                record.student.student_name, record.student.roll_number, record.student.registration_number,
                record.student.subject_name, record.student.subject_code, record.student.exam_name, record.student.exam_date,
                record.student.college_name, record.student.semester, record.student.branch, record.total_max_marks,
                record.ai_total_marks, record.final_total_marks, record.percentage, record.grade,
                1 if record.is_approved else 0, record.id
            ))

            cursor.execute("DELETE FROM question_evaluations WHERE evaluation_id=?", (record.id,))
            for q in record.questions:
                cursor.execute("""
                    INSERT INTO question_evaluations (
                        evaluation_id, question_id, student_answer, model_answer,
                        max_marks, ai_marks, final_marks, similarity_score,
                        matched_keywords, missing_keywords, feedback, teacher_comments
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.id, q.question_id, q.student_answer, q.model_answer,
                    q.max_marks, q.ai_marks, q.final_marks, q.similarity_score,
                    json.dumps(q.matched_keywords), json.dumps(q.missing_keywords),
                    q.feedback, q.teacher_comments
                ))
            conn.commit()
            return True

    def get_evaluation(self, eval_id: int) -> Optional[EvaluationRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evaluations WHERE id=?", (eval_id,))
            row = cursor.fetchone()
            if not row:
                return None

            student = StudentDetails(
                student_name=row["student_name"],
                roll_number=row["roll_number"],
                registration_number=row["registration_number"],
                subject_name=row["subject_name"],
                subject_code=row["subject_code"],
                exam_name=row["exam_name"],
                exam_date=row["exam_date"],
                college_name=row["college_name"],
                semester=row["semester"],
                branch=row["branch"]
            )

            cursor.execute("SELECT * FROM question_evaluations WHERE evaluation_id=?", (eval_id,))
            q_rows = cursor.fetchall()
            questions = []
            for qr in q_rows:
                questions.append(QuestionEvaluation(
                    question_id=qr["question_id"],
                    student_answer=qr["student_answer"],
                    model_answer=qr["model_answer"],
                    max_marks=qr["max_marks"],
                    ai_marks=qr["ai_marks"],
                    final_marks=qr["final_marks"],
                    similarity_score=qr["similarity_score"],
                    matched_keywords=json.loads(qr["matched_keywords"] or "[]"),
                    missing_keywords=json.loads(qr["missing_keywords"] or "[]"),
                    feedback=qr["feedback"],
                    teacher_comments=qr["teacher_comments"]
                ))

            return EvaluationRecord(
                id=row["id"],
                booklet_filename=row["booklet_filename"],
                student=student,
                total_max_marks=row["total_max_marks"],
                ai_total_marks=row["ai_total_marks"],
                final_total_marks=row["final_total_marks"],
                percentage=row["percentage"],
                grade=row["grade"],
                is_approved=bool(row["is_approved"]),
                evaluation_date=row["evaluation_date"],
                questions=questions
            )

    def search_evaluations(
        self,
        query: str = "",
        subject_filter: str = "All",
        exam_filter: str = "All",
        grade_filter: str = "All",
        status_filter: str = "All",
        sort_by: str = "Date"
    ) -> List[EvaluationRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            sql = "SELECT id, percentage FROM evaluations WHERE 1=1"
            params = []

            if query.strip():
                sql += " AND (student_name LIKE ? OR roll_number LIKE ? OR registration_number LIKE ?)"
                pat = f"%{query.strip()}%"
                params.extend([pat, pat, pat])

            if subject_filter and subject_filter != "All":
                sql += " AND (subject_name = ? OR subject_code = ?)"
                params.extend([subject_filter, subject_filter])

            if exam_filter and exam_filter != "All":
                sql += " AND exam_name = ?"
                params.append(exam_filter)

            if grade_filter and grade_filter != "All":
                sql += " AND grade = ?"
                params.append(grade_filter)

            settings = self.get_settings()
            pass_thresh = settings.pass_percentage

            if status_filter == "PASS":
                sql += " AND percentage >= ?"
                params.append(pass_thresh)
            elif status_filter == "FAIL":
                sql += " AND percentage < ?"
                params.append(pass_thresh)

            # Sorting
            if sort_by == "Name":
                sql += " ORDER BY student_name ASC"
            elif sort_by == "Marks":
                sql += " ORDER BY final_total_marks DESC"
            elif sort_by == "Percentage":
                sql += " ORDER BY percentage DESC"
            elif sort_by == "Grade":
                sql += " ORDER BY grade ASC"
            else:
                sql += " ORDER BY id DESC"

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            results = []
            for r in rows:
                ev = self.get_evaluation(r["id"])
                if ev:
                    results.append(ev)
            return results

    def get_dashboard_stats(self) -> Dict[str, Any]:
        return self.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        settings = self.get_settings()
        pass_thresh = settings.pass_percentage

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(percentage), MAX(final_total_marks), MIN(final_total_marks) FROM evaluations")
            count, avg_pct, max_marks, min_marks = cursor.fetchone()
            
            if not count or count == 0:
                return {
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

            cursor.execute("SELECT COUNT(*) FROM evaluations WHERE percentage >= ?", (pass_thresh,))
            pass_count = cursor.fetchone()[0]
            fail_count = count - pass_count
            pass_pct = (pass_count / count) * 100.0 if count > 0 else 0.0

            cursor.execute("SELECT grade, COUNT(*) FROM evaluations GROUP BY grade")
            grade_counts = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("""
                SELECT subject_name, COUNT(*), AVG(percentage), MAX(percentage), MIN(percentage)
                FROM evaluations
                GROUP BY subject_name
            """)
            subject_stats = []
            for row in cursor.fetchall():
                subject_stats.append({
                    "subject": row[0],
                    "count": row[1],
                    "avg_percentage": round(row[2] or 0.0, 2),
                    "max_percentage": round(row[3] or 0.0, 2),
                    "min_percentage": round(row[4] or 0.0, 2)
                })

            return {
                "total_students": count,
                "total_passed": pass_count,
                "total_failed": fail_count,
                "avg_percentage": round(avg_pct or 0.0, 2),
                "pass_percentage": round(pass_pct, 2),
                "highest_marks": round(max_marks or 0.0, 2),
                "lowest_marks": round(min_marks or 0.0, 2),
                "grade_counts": grade_counts,
                "subject_stats": subject_stats
            }

    def get_all_subjects(self) -> List[Dict[str, str]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT subject_code, subject_name FROM model_answers")
            rows = cursor.fetchall()
            if not rows:
                return [{"subject_code": "CS501", "subject_name": "Machine Learning"}]
            return [{"subject_code": r["subject_code"], "subject_name": r["subject_name"]} for r in rows]

    def get_all_exams(self) -> List[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT exam_name FROM evaluations")
            rows = cursor.fetchall()
            return [r["exam_name"] for r in rows if r["exam_name"]]

    # -------------------------------------------------------------
    # Field Verifications Persistent Storage
    # -------------------------------------------------------------
    def save_field_verifications(
        self,
        student_uid: str,
        booklet_filename: str,
        v_map: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        roll_number: Optional[str] = None,
        evaluation_id: Optional[int] = None
    ) -> bool:
        """Saves or updates field verification statuses and edited values for a student in SQLite."""
        if not student_uid:
            return False
        with self._get_connection() as conn:
            cursor = conn.cursor()
            field_data_json = json.dumps(v_map)
            acc = float(metrics.get("accuracy_pct", 0.0)) if metrics else 0.0
            err = float(metrics.get("error_pct", 0.0)) if metrics else 0.0
            pen = float(metrics.get("pending_pct", 100.0)) if metrics else 100.0
            v_cnt = int(metrics.get("verified_correct_count", 0)) if metrics else 0
            i_cnt = int(metrics.get("incorrect_count", 0)) if metrics else 0
            p_cnt = int(metrics.get("needs_verification_count", 11)) if metrics else 11

            cursor.execute("""
                INSERT INTO field_verifications (
                    student_uid, booklet_filename, roll_number, field_data,
                    accuracy_pct, error_pct, pending_pct, verified_count,
                    incorrect_count, pending_count, evaluation_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(student_uid) DO UPDATE SET
                    booklet_filename=excluded.booklet_filename,
                    roll_number=COALESCE(excluded.roll_number, field_verifications.roll_number),
                    field_data=excluded.field_data,
                    accuracy_pct=excluded.accuracy_pct,
                    error_pct=excluded.error_pct,
                    pending_pct=excluded.pending_pct,
                    verified_count=excluded.verified_count,
                    incorrect_count=excluded.incorrect_count,
                    pending_count=excluded.pending_count,
                    evaluation_id=COALESCE(excluded.evaluation_id, field_verifications.evaluation_id),
                    updated_at=CURRENT_TIMESTAMP
            """, (
                student_uid, booklet_filename, roll_number, field_data_json,
                acc, err, pen, v_cnt, i_cnt, p_cnt, evaluation_id
            ))
            conn.commit()
            return True

    def get_field_verifications(self, student_uid: str, roll_number: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves saved field verification data and metrics for a student from SQLite."""
        if not student_uid:
            return None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM field_verifications WHERE student_uid=?", (student_uid,))
            row = cursor.fetchone()
            if not row and roll_number:
                cursor.execute(
                    "SELECT * FROM field_verifications WHERE roll_number=? ORDER BY updated_at DESC LIMIT 1",
                    (roll_number,)
                )
                row = cursor.fetchone()
            if not row:
                return None
            try:
                fdata = json.loads(row["field_data"] or "{}")
            except Exception:
                fdata = {}
            return {
                "student_uid": row["student_uid"],
                "booklet_filename": row["booklet_filename"],
                "roll_number": row["roll_number"],
                "field_data": fdata,
                "accuracy_pct": row["accuracy_pct"],
                "error_pct": row["error_pct"],
                "pending_pct": row["pending_pct"],
                "verified_count": row["verified_count"],
                "incorrect_count": row["incorrect_count"],
                "pending_count": row["pending_count"],
                "evaluation_id": row["evaluation_id"],
                "updated_at": row["updated_at"]
            }
