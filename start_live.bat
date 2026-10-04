@echo off
cd /d %~dp0
echo ============================================
echo   Deepfake Detection System - LIVE LINK
echo ============================================
echo.
echo [1/2] Starting Streamlit on http://localhost:8501 ...
start "Streamlit - Deepfake Detection" cmd /k "python -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false"
timeout /t 15 /nobreak >nul
echo.
echo [2/2] Starting public tunnel - copy the URL it prints below:
echo ------------------------------------------------------------
npx --yes localtunnel --port 8501
echo ------------------------------------------------------------
echo Keep this window OPEN while you share the link.
echo (If the link stops working, close both windows and run this again.)
pause