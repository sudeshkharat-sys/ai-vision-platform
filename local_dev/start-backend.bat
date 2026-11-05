@echo off
call "%~dp0_config.bat"
call "%CONDA_ACTIVATE%" %CONDA_ENV%
cd /d "%PROJ%\backend"
echo [%PROJECT_NAME%] backend on http://127.0.0.1:%BACKEND_PORT%
python -m uvicorn %BACKEND_APP% --host 127.0.0.1 --port %BACKEND_PORT% --reload
