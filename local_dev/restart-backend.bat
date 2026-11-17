@echo off
rem Stops whatever listens on the backend port, then starts the backend in THIS window.
call "%~dp0_config.bat"
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%BACKEND_PORT% .*LISTENING"') do taskkill /PID %%P /T /F >nul 2>&1
timeout /t 1 >nul
call "%~dp0start-backend.bat"
