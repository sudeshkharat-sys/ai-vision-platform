@echo off
rem Stops running Celery workers of this project, then starts one in THIS window.
call "%~dp0_config.bat"
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'celery' -and $_.CommandLine -match '%CELERY_APP%' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
timeout /t 1 >nul
call "%~dp0start-celery.bat"
