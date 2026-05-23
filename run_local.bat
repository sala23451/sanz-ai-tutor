@echo off
:: Sanz AI Tutor Local Launch Script
:: Set your real Gemini API key below to enable AI tutoring features
set GOOGLE_API_KEY=dummy-key
echo Starting Sanz AI Tutor local server...
.venv\Scripts\python.exe -m backend.main
pause
