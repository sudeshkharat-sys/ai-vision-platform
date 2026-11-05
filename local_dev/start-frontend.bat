@echo off
call "%~dp0_config.bat"
cd /d "%PROJ%\frontend"
if not exist node_modules (
  echo First run: installing frontend packages...
  call npm install
)
if defined FRONTEND_PORT set "PORT=%FRONTEND_PORT%"
npm start
