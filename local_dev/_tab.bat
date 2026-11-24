@echo off
rem Runs one service inside an Anaconda-Prompt-style shell: activate conda base, then the service script.
rem Usage: _tab.bat redis|backend|celery|frontend   (no && in start-all.bat, so no quoting problems)
call "%~dp0_config.bat"
if defined CONDA_ACTIVATE call "%CONDA_ACTIVATE%"
title Anaconda Prompt - %PROJECT_NAME% %1
call "%~dp0start-%1.bat"
rem Service ended/crashed: stay in this Anaconda Prompt (env active) so you can read the output and rerun.
echo.
echo [%1 stopped] You are in an Anaconda Prompt (env: %CONDA_ENV%). Rerun with the %PROJECT_SHORT%-%1 command.

