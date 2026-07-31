@echo off
title TTAL Transcriber & Dubbing Studio
echo ========================================================
echo   Launching TTAL Video Transcriber & Dubbing Studio
echo   Target GPU: NVIDIA GeForce GTX 1060 (6GB VRAM)
echo ========================================================
echo.

set VENV_DIR=%~dp0.venv
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe

rem Check & Start Ollama if installed
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    echo [Ollama] Checking local LLM background daemon...
    netstat -ano | findstr :11434 >nul 2>&1
    if %errorlevel% neq 0 (
        echo [Ollama] Starting Ollama local server in background...
        start /b ollama serve >nul 2>&1
        timeout /t 3 >nul
    )
) else (
    echo [Ollama] Note: Ollama installation is downloading in background.
)

if not exist "%PYTHON_EXE%" (
    echo [1/3] Creating Python 3.12 Virtual Environment...
    py -3.12 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Python 3.12 not found. Falling back to default python...
        python -m venv "%VENV_DIR%"
    )
)

echo [2/3] Checking dependencies...
call "%VENV_DIR%\Scripts\activate.bat"
"%PYTHON_EXE%" -m pip install -q -r requirements.txt

echo.
echo [3/3] Starting Local Web Application server...
echo Server running at http://127.0.0.1:8000
echo Opening browser in 3 seconds...
echo (Keep this window open while using the app)
echo.

start /b cmd /c "timeout /t 3 >nul && start http://127.0.0.1:8000"

"%PYTHON_EXE%" app.py

pause
