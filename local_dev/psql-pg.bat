@echo off
call "%~dp0_config.bat"
"%PGBIN%\psql.exe" -U postgres -W
