@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_google_drive.ps1"
if errorlevel 1 (
  echo.
  echo Setup did not finish. Keep this window open and send Codex a screenshot.
) else (
  echo.
  echo Google Drive setup finished successfully.
)
echo.
pause
