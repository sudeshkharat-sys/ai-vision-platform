@echo off
call "%~dp0_config.bat"
cd /d "%REDISDIR%"
echo Starting Redis. Keep this window open.
rem No persistence: stale queued/cancelled tasks must NOT come back after a restart.
redis-server.exe --save "" --appendonly no
