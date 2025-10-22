@echo off
REM Wrapper to run PowerShell script

REM Ensure PowerShell execution policy allows script execution
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0\download_models.ps1"
pause