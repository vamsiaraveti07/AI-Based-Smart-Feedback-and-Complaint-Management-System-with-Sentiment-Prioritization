@echo off
echo ====================================================
echo 🎯 Enhanced AI-Powered Grievance System
echo ====================================================
echo.
echo Starting the system...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Run the startup script
python run_system.py

echo.
echo Press any key to exit...
pause >nul
