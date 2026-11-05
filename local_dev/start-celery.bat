@echo off
call "%~dp0_config.bat"
call "%CONDA_ACTIVATE%" %CONDA_ENV%
cd /d "%PROJ%\backend"
echo [%PROJECT_NAME%] Celery worker (Redis must be running first).
celery -A %CELERY_APP% worker --loglevel=info --pool=solo -Q celery
