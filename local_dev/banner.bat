@echo off
rem Run at the start of a terminal tab: puts the short commands on PATH for this window and prints the list.
set "PATH=%USERPROFILE%\dev-tools;%PATH%"
echo.
echo AI-VISION COMMANDS READY:
echo   pg-start / pg-stop / pg-psql     (Postgres, port 5432)
echo   redis-start                      (Redis, port 6379)
echo   vi-backend / vi-celery / vi-frontend
echo   vi-all                           (everything, each in its own window)
echo   save-history                     (saves this window's commands to a file)
echo.
echo (If a command is not found, run local_dev\install-shortcuts.bat once.)
echo.
