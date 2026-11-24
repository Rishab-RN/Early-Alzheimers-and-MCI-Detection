@echo off
REM Starts the local research dashboard at http://127.0.0.1:8000
REM   run_dashboard.bat         aggregate mode (safe to screenshot)

cd /d "%~dp0"
set OTEL_SDK_DISABLED=true
if /I "%1"=="local" (set ANN_LOCAL_DATA=1) else (set ANN_LOCAL_DATA=0)
if not exist app\web\dist\index.html echo Frontend not built. See app\README.md && exit /b 1
start "" http://127.0.0.1:8000
.venv-app\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000
