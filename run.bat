@echo off
echo =====================================================================
echo  Credit Risk Decisioning & Model Risk Evaluation Platform
echo  Starting Flask Server on Python 3.12...
echo =====================================================================
echo.

where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3.12 app.py
) else (
    python app.py
)

pause
