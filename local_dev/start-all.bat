@echo off
rem One command: Postgres, Redis, backend, Celery, frontend.
rem With Windows Terminal (wt.exe): ONE window named after the project, one tab per service.
rem Without it: a separate window per service.
call "%~dp0_config.bat"
call "%~dp0start-postgres.bat"
where wt >nul 2>&1
if errorlevel 1 goto separate

rem Same window name each time => every call adds a tab to that one window.
wt -w "%PROJECT_NAME%" new-tab --suppressApplicationTitle --title "Redis" cmd /k "%~dp0start-redis.bat"
timeout /t 2 >nul
wt -w "%PROJECT_NAME%" new-tab --suppressApplicationTitle --title "Backend" cmd /k "%~dp0start-backend.bat"
timeout /t 1 >nul
wt -w "%PROJECT_NAME%" new-tab --suppressApplicationTitle --title "Celery" cmd /k "%~dp0start-celery.bat"
timeout /t 1 >nul
wt -w "%PROJECT_NAME%" new-tab --suppressApplicationTitle --title "Frontend" cmd /k "%~dp0start-frontend.bat"
goto :eof

:separate
echo Windows Terminal (wt) not found - opening separate windows.
start "%PROJECT_NAME% Redis" cmd /k "%~dp0start-redis.bat"
timeout /t 3 >nul
start "%PROJECT_NAME% Backend" cmd /k "%~dp0start-backend.bat"
start "%PROJECT_NAME% Celery" cmd /k "%~dp0start-celery.bat"
start "%PROJECT_NAME% Frontend" cmd /k "%~dp0start-frontend.bat"
