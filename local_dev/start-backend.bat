@echo off
call "%~dp0_config.bat"
call "%CONDA_ACTIVATE%" %CONDA_ENV%
cd /d "%PROJ%\backend"
echo [%PROJECT_NAME%] backend on http://127.0.0.1:%BACKEND_PORT%
echo Log file: %PROJ%\backend.log  (read the last lines after a crash)
rem PYTHONFAULTHANDLER prints a traceback even on a hard crash (CUDA / native library).
rem --reload-dir app: only restart when code in app\ changes, not on data/model/upload files.
set PYTHONFAULTHANDLER=1
set PYTHONUNBUFFERED=1
rem Auto-reload is OFF by default: a reload during training/long jobs restarts the API (re-login, dropped
rem in-process jobs). Use vi-restart after code changes, or set BACKEND_RELOAD=1 to get --reload back.
set "RELOAD_ARGS="
if "%BACKEND_RELOAD%"=="1" set "RELOAD_ARGS=--reload --reload-dir app"
rem 2>&1 is done by cmd (not PowerShell) so uvicorn's normal INFO lines are not shown as red NativeCommandError.
python -X faulthandler -m uvicorn %BACKEND_APP% --host 127.0.0.1 --port %BACKEND_PORT% %RELOAD_ARGS% 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%PROJ%\backend.log'"
echo.
echo Backend stopped. Last lines of the log:
powershell -NoProfile -Command "Get-Content '%PROJ%\backend.log' -Tail 40"
