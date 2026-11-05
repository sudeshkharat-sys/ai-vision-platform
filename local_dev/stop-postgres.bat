@echo off
call "%~dp0_config.bat"
"%PGBIN%\pg_ctl.exe" stop -D "%PGDATA%"
