# 🎓 SMART AI EXAMINATION EVALUATOR

> An end-to-end AI-powered examination evaluation web application featuring automatic first-page student metadata extraction, OpenCV image preprocessing, Tesseract OCR, SentenceTransformers NLP semantic grading, interactive teacher review, and ReportLab PDF marksheets.

---

## 🛑 IMPORTANT: HOW TO RUN THE APPLICATION

### ❌ DO NOT RUN:
```powershell
python app.py
```
*(Running `python app.py` directly without Streamlit will cause `ModuleNotFoundError: No module named 'streamlit'`).*

---

### ✅ CORRECT WAY TO RUN:

#### Option 1: Double-Click Batch File (Easiest for Beginners!)
Simply double-click **`run_app.bat`** in the project folder! It will automatically set up the virtual environment, install requirements, and open the app in your web browser.

---

#### Option 2: Windows PowerShell Step-by-Step Guide

**STEP 1: Open PowerShell in the project folder.**

**STEP 2: Create a virtual environment**
```powershell
python -m venv .venv
```

**STEP 3: Upgrade pip**
```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
```

**STEP 4: Install dependencies**
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**STEP 5: Launch the Streamlit Web Application**
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

**STEP 6: Open Browser**
Open `http://localhost:8501` in your browser.

> **Note on PowerShell Script Blocking**: If PowerShell displays an error about `Activate.ps1` script execution policies, you do NOT need to activate the environment! Using `.venv\Scripts\python.exe -m streamlit run app.py` bypasses PowerShell script restriction policies completely.

---

## 🔤 TESSERACT OCR INSTALLATION (OPTIONAL FOR OPTICAL SCANNING)

`pytesseract` is a Python wrapper for the Tesseract OCR engine binary.

### Installing Tesseract OCR on Windows:
1. Download the Windows installer from [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
2. Run the installer (Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`).
3. Verify installation in PowerShell:
   ```powershell
   tesseract --version
   ```
4. If Tesseract is not in your system PATH, configure its path in `.env` or in the **Settings** menu tab:
   ```env
   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

*Note: If Tesseract binary is missing, the application will display a clear warning banner and fallback gracefully to layout parsing mode without crashing.*

---

## 🔑 DEMO LOGIN CREDENTIALS

- **Teacher Portal**: `teacher@smartai.edu` / `teacher123`
- **Admin Portal**: `admin@smartai.edu` / `admin123`

---

## 🌟 KEY FEATURES & WORKFLOW

1. **Upload Booklet**: Upload PDF, JPG, JPEG, or PNG. Unique filenames saved in `data/uploads/`.
2. **First Page OCR**: OpenCV computer vision pipeline extracts 10 student metadata fields (*Name, Roll No, Reg No, Subject, Code, Exam Name, Date, College, Sem, Branch*).
3. **Teacher Verification**: Verify and edit extracted details in editable text boxes.
4. **Answer Extraction**: Detects question markers (`Q1`, `Q2`...) across answer pages.
5. **Side-by-Side Review**: Original page snippet alongside editable extracted answer.
6. **AI Similarity Evaluation**: Cosine similarity & keyword coverage matching against model answers.
7. **Teacher Override & Approval**: AI suggested marks are marked as suggestions (*"AI-generated marks are suggestions. Please verify before approving"*). Official marks are saved only upon clicking **`[ APPROVE FINAL RESULT ]`**.
8. **All Students Results**: Dynamic database table of evaluated booklets with Multi-Student isolation.
9. **Reports & Exports**: 1-click **Download PDF Marksheet** (ReportLab), **Export to Excel (`.xlsx`)**, and **Export to CSV**.

---

## 📄 LICENSE
Created for university examination evaluation systems. All rights reserved.
