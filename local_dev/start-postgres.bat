@echo off
call "%~dp0_config.bat"
"%PGBIN%\pg_ctl.exe" status -D "%PGDATA%" >nul 2>&1
if %errorlevel%==0 (
  echo PostgreSQL is already running.
) else (
  "%PGBIN%\pg_ctl.exe" start -D "%PGDATA%" -l "%PGLOG%"
)
