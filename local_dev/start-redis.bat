@echo off
call "%~dp0_config.bat"
cd /d "%REDISDIR%"
echo Starting Redis. Keep this window open.
rem Old snapshots are LOADED on start, so delete them or old tasks come back.
del /q dump.rdb >nul 2>&1
rmdir /s /q appendonlydir >nul 2>&1
rem No persistence: stale queued/cancelled tasks must NOT come back after a restart.
redis-server.exe --save "" --appendonly no
