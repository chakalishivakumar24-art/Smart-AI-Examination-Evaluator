from database.database import EvaluationDatabase
from models.database_models import StudentDetails, QuestionEvaluation, EvaluationRecord, ModelAnswerItem

def seed_sample_model_answers(db: EvaluationDatabase):
    """Seeds sample model answers for Machine Learning (CS501)."""
    items = [
        ModelAnswerItem(
            question_id="Q1",
            question_text="Define Supervised Learning and explain the difference between Classification and Regression.",
            model_answer="Supervised learning is a machine learning paradigm where algorithms are trained using labeled datasets. Classification predicts discrete categorical class labels (e.g., Spam or Not Spam), whereas regression predicts continuous numerical values (e.g., house price or temperature prediction). Key algorithms include Decision Trees, SVM, Linear Regression, and Logistic Regression.",
            max_marks=10.0,
            keywords=["supervised learning", "labeled dataset", "classification", "regression", "discrete", "continuous", "decision trees"]
        ),
        ModelAnswerItem(
            question_id="Q2",
            question_text="What is Overfitting in Machine Learning models and how can it be prevented?",
            model_answer="Overfitting occurs when a machine learning model learns the training data too well, including its noise and outliers, leading to high training accuracy but poor generalization performance on unseen test data. It can be prevented using techniques such as regularization (L1/L2), cross-validation, pruning decision trees, dropout in neural networks, and acquiring more training data.",
            max_marks=10.0,
            keywords=["overfitting", "training data", "generalization", "unseen test data", "regularization", "cross-validation", "dropout", "pruning"]
        ),
        ModelAnswerItem(
            question_id="Q3",
            question_text="Explain the working principle of Support Vector Machines (SVM) and the role of Hyperplanes.",
            model_answer="Support Vector Machine (SVM) is a supervised algorithm used for classification and regression. SVM finds an optimal hyperplane that maximizes the margin (distance) between different classes of data points in an N-dimensional space. Points closest to the boundary are called support vectors. SVM uses kernel tricks (RBF, polynomial) to handle non-linearly separable data.",
            max_marks=10.0,
            keywords=["support vector machine", "svm", "hyperplane", "margin", "support vectors", "classification", "kernel trick"]
        ),
        ModelAnswerItem(
            question_id="Q4",
            question_text="What are Confusion Matrices and how are Precision and Recall calculated?",
            model_answer="A confusion matrix is an NxN table used to evaluate performance of a classification model by summarizing True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). Precision is calculated as TP / (TP + FP) measuring accuracy of positive predictions. Recall is calculated as TP / (TP + FN) measuring coverage of actual positive cases.",
            max_marks=10.0,
            keywords=["confusion matrix", "true positive", "true negative", "false positive", "false negative", "precision", "recall", "accuracy"]
        )
    ]
    db.save_model_answers(subject_code="CS501", subject_name="Machine Learning", items=items)

def seed_demo_evaluations(db: EvaluationDatabase):
    """Seeds sample student evaluation records matching the exact example provided in requirements."""
    # Seed model answers first
    seed_sample_model_answers(db)

    # Check if demo records already exist
    existing = db.search_evaluations()
    if len(existing) > 0:
        return

    # Record 1: Rahul Kumar (Exact example from requirement prompt)
    student1 = StudentDetails(
        student_name="Rahul Kumar",
        roll_number="21CS045",
        registration_number="REG2026CS045",
        subject_name="Machine Learning",
        subject_code="CS501",
        exam_name="Semester Examination",
        exam_date="20-08-2026",
        college_name="ABC Engineering College",
        semester="V",
        branch="Computer Science"
    )

    q1 = QuestionEvaluation(
        question_id="Q1",
        student_answer="Supervised learning uses labeled training data. Classification predicts discrete output labels like pass/fail, while regression predicts continuous numbers such as stock prices. Examples include decision trees and linear regression.",
        model_answer="Supervised learning is a machine learning paradigm where algorithms are trained using labeled datasets. Classification predicts discrete categorical class labels, whereas regression predicts continuous numerical values.",
        max_marks=10.0,
        ai_marks=8.0,
        final_marks=8.0,
        similarity_score=0.86,
        matched_keywords=["supervised learning", "labeled training data", "classification", "regression", "discrete", "continuous"],
        missing_keywords=["decision trees", "svm"],
        feedback="Good answer with correct distinction between classification and regression.",
        teacher_comments="Accurate concepts."
    )

    q2 = QuestionEvaluation(
        question_id="Q2",
        student_answer="Overfitting happens when the model memorizes the training data noise. It gives 100% training accuracy but fails on test data. We can prevent it using cross-validation and adding more data.",
        model_answer="Overfitting occurs when a machine learning model learns training data noise, leading to poor generalization on unseen data. Prevented via regularization (L1/L2), cross-validation, pruning, dropout.",
        max_marks=10.0,
        ai_marks=7.0,
        final_marks=7.0,
        similarity_score=0.74,
        matched_keywords=["overfitting", "training data", "noise", "cross-validation"],
        missing_keywords=["regularization", "dropout", "pruning"],
        feedback="Partially complete. Missing important concept: Regularization techniques (L1/L2) and Dropout.",
        teacher_comments="Mention regularization next time."
    )

    q3 = QuestionEvaluation(
        question_id="Q3",
        student_answer="Support Vector Machine (SVM) finds the optimal hyperplane to separate classes with maximum margin. Data points closest to decision boundary are called support vectors. Kernel trick is used for non-linear boundaries.",
        model_answer="SVM finds optimal hyperplane maximizing margin between classes in N-dimensional space. Support vectors are nearest points. Uses kernel trick for non-linear data.",
        max_marks=10.0,
        ai_marks=9.0,
        final_marks=9.0,
        similarity_score=0.92,
        matched_keywords=["support vector machine", "svm", "hyperplane", "margin", "support vectors", "kernel trick"],
        missing_keywords=[],
        feedback="Excellent answer covering all core concepts and terminology.",
        teacher_comments="Outstanding explanation."
    )

    q4 = QuestionEvaluation(
        question_id="Q4",
        student_answer="Confusion matrix summarizes TP, TN, FP, FN. Precision formula is TP/(TP+FP). Recall formula is TP/(TP+FN).",
        model_answer="Confusion matrix summarizes TP, TN, FP, FN. Precision = TP / (TP + FP). Recall = TP / (TP + FN).",
        max_marks=10.0,
        ai_marks=6.0,
        final_marks=6.0,
        similarity_score=0.68,
        matched_keywords=["confusion matrix", "tp", "tn", "fp", "fn", "precision", "recall"],
        missing_keywords=["accuracy of positive predictions", "coverage of actual positive cases"],
        feedback="Partially correct. Included formulas but missing intuitive definitions of precision and recall.",
        teacher_comments="Add conceptual explanation alongside formulas."
    )

    rec1 = EvaluationRecord(
        booklet_filename="Rahul_Kumar_21CS045_Booklet.pdf",
        student=student1,
        total_max_marks=15.0,
        ai_total_marks=30.0,
        final_total_marks=30.0,
        percentage=75.0,
        grade="A",
        is_approved=True,
        evaluation_date="2026-08-20 14:30:00",
        questions=[q1, q2, q3, q4]
    )
    db.save_evaluation(rec1)

    # Record 2: Priya Sharma
    student2 = StudentDetails(
        student_name="Priya Sharma",
        roll_number="21CS052",
        registration_number="REG2026CS052",
        subject_name="Machine Learning",
        subject_code="CS501",
        exam_name="Semester Examination",
        exam_date="20-08-2026",
        college_name="ABC Engineering College",
        semester="V",
        branch="Computer Science"
    )

    p1 = QuestionEvaluation(
        question_id="Q1",
        student_answer="Supervised learning requires labeled dataset. Classification outputs category labels, regression outputs numbers.",
        model_answer="Supervised learning trains on labeled dataset. Classification predicts discrete classes, regression predicts continuous values.",
        max_marks=10.0, ai_marks=9.0, final_marks=9.0, similarity_score=0.90,
        matched_keywords=["supervised learning", "labeled dataset", "classification", "regression"],
        missing_keywords=[], feedback="Very good explanation.", teacher_comments="Correct."
    )
    p2 = QuestionEvaluation(
        question_id="Q2",
        student_answer="Overfitting is when model fits noise. Prevented using L1/L2 regularization, dropout, and cross validation.",
        model_answer="Overfitting is model learning noise. Prevented via regularization, dropout, cross validation.",
        max_marks=10.0, ai_marks=10.0, final_marks=10.0, similarity_score=0.96,
        matched_keywords=["overfitting", "noise", "regularization", "dropout", "cross validation"],
        missing_keywords=[], feedback="Complete answer with all key terms.", teacher_comments="Full marks."
    )
    rec2 = EvaluationRecord(
        booklet_filename="Priya_Sharma_21CS052_Booklet.pdf",
        student=student2,
        total_max_marks=20.0, ai_total_marks=19.0, final_total_marks=19.0,
        percentage=95.0, grade="A+", is_approved=True,
        evaluation_date="2026-08-21 10:15:00", questions=[p1, p2]
    )
    db.save_evaluation(rec2)
