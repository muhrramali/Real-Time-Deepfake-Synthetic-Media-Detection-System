@echo off
setlocal
cd /d %~dp0
echo ============================================
echo   Deepfake Detection System - Hackathon Run
echo ============================================
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found on PATH.
  exit /b 1
)
echo [1/2] Installing requirements...
python -m pip install -r requirements.txt
echo [2/2] Launching Streamlit dashboard...
streamlit run app.py
endlocal