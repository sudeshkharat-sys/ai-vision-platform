@echo off
call "%~dp0_config.bat"
powershell -NoProfile -Command "Get-Content '%PROJ%\backend.log' -Tail 60"
