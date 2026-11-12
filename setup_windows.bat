@echo off
title Smart AI Examination Evaluator - Windows Setup
echo ===================================================
echo   SMART AI EXAMINATION EVALUATOR - WINDOWS SETUP
echo ===================================================
echo.

:: Step 1: Check Python Installation & Version
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not added to your system PATH.
    echo Please download and install Python 3.9 - 3.12 from https://www.python.org/
    echo Make sure to check the box "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] System Python detected:
python --version

:: Step 2: Create Virtual Environment
echo.
echo [1/5] Creating Python Virtual Environment (.venv)...
if not exist .venv (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)
echo [OK] Virtual environment ready (.venv).

:: Step 3: Upgrade Pip
echo.
echo [2/5] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

:: Step 4: Install Requirements
echo.
echo [3/5] Installing required packages from requirements.txt...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some packages reported warnings. Continuing verification...
)

:: Step 5: Verify & Auto-Install Streamlit
echo.
echo [4/5] Verifying Streamlit installation...
.venv\Scripts\python.exe -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [NOTICE] Streamlit missing in .venv. Installing Streamlit automatically...
    .venv\Scripts\python.exe -m pip install streamlit
)

.venv\Scripts\python.exe -m streamlit --version >nul 2>&1
if errorlevel 0 (
    echo [OK] Streamlit installation verified successfully!
) else (
    echo [ERROR] Streamlit installation failed. Retrying...
    .venv\Scripts\python.exe -m pip install --no-cache-dir streamlit
)

:: Step 6: Environment Diagnostic Verification
echo.
echo [5/5] Running environment diagnostic check...
.venv\Scripts\python.exe check_environment.py

echo.
echo ===================================================
echo  SETUP PROCESS COMPLETE!
echo.
echo  To launch the application, you can:
echo    1. Double-click "run_app.bat"
echo    OR
echo    2. Run: .venv\Scripts\python.exe -m streamlit run app.py
echo ===================================================
echo.
pause
