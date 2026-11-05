@echo off
call "%~dp0_config.bat"
cd /d "%REDISDIR%"
echo Starting Redis. Keep this window open.
redis-server.exe
