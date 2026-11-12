@echo off
title Smart AI Examination Evaluator Launcher
echo ===================================================
echo   LAUNCHING SMART AI EXAMINATION EVALUATOR
echo ===================================================
echo.

:: Check if .venv exists; run setup if missing
if not exist .venv (
    echo [NOTICE] Virtual environment .venv not found. Running setup_windows.bat first...
    call setup_windows.bat
)

:: Verify Streamlit inside .venv; install if missing
.venv\Scripts\python.exe -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [NOTICE] Streamlit is missing in .venv. Installing Streamlit automatically...
    .venv\Scripts\python.exe -m pip install streamlit
)

echo Starting Streamlit Web Application...
echo Local URL: http://localhost:8501
echo.
.venv\Scripts\python.exe -m streamlit run app.py

if errorlevel 1 (
    echo.
    echo ===================================================
    echo [ERROR] Application failed to launch or was stopped.
    echo.
    echo Suggested Fixes:
    echo  1. Run setup_windows.bat to re-install requirements.
    echo  2. Manually run: .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo  3. Launch using: .venv\Scripts\python.exe -m streamlit run app.py
    echo ===================================================
    pause
)
