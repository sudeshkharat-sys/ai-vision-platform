@echo off
rem Run once per project. Creates short typed commands in %USERPROFILE%\dev-tools:
rem   pg-start  pg-stop  pg-psql  redis-start  save-history
rem   <prefix>-all  <prefix>-backend  <prefix>-celery  <prefix>-frontend  <prefix>-install
call "%~dp0_config.bat"
set "HERE=%~dp0"
set "OUT=%USERPROFILE%\dev-tools"
set "P=%PROJECT_SHORT%"
if not exist "%OUT%" mkdir "%OUT%"
call :w pg-start start-postgres
call :w pg-stop stop-postgres
call :w pg-psql psql-pg
call :w redis-start start-redis
call :w save-history save-history
call :w %P%-all start-all
call :w %P%-backend start-backend
call :w %P%-celery start-celery
call :w %P%-frontend start-frontend
call :w %P%-install install-requirements
echo.
echo Created commands in %OUT%
echo.
echo ONE-TIME: add that folder to your user PATH so they work in every window:
echo   Start menu ^> type "environment variables" ^> "Edit environment variables for your account"
echo   ^> select Path ^> Edit ^> New ^> paste  %OUT%  ^> OK
echo Then open a NEW Anaconda Prompt and type e.g.  %P%-all
exit /b 0
:w
> "%OUT%\%1.bat" echo @call "%HERE%%2.bat" %%*
exit /b 0
