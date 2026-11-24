@echo off
call "%~dp0_config.bat"
call "%CONDA_ACTIVATE%" %CONDA_ENV%
cd /d "%PROJ%\backend"
echo [%PROJECT_NAME%] Celery worker (Redis must be running first).
rem Wipe the broker DB (queued + unacked + revoked) so crashed/cancelled tasks never re-run.
python -c "import redis; redis.Redis.from_url('redis://localhost:6379/1').flushdb(); print('broker cleared')"
python -m celery -A %CELERY_APP% worker --loglevel=info --pool=solo -Q celery --purge
