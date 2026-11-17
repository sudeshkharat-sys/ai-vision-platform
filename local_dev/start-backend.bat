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
powershell -NoProfile -Command "python -X faulthandler -m uvicorn %BACKEND_APP% --host 127.0.0.1 --port %BACKEND_PORT% --reload --reload-dir app 2>&1 | Tee-Object -FilePath '%PROJ%\backend.log'"
echo.
echo Backend stopped. Last lines of the log:
powershell -NoProfile -Command "Get-Content '%PROJ%\backend.log' -Tail 40"
