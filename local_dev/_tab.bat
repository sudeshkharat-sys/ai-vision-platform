@echo off
rem Runs one service inside an Anaconda-Prompt-style shell: activate conda base, then the service script.
rem Usage: _tab.bat redis|backend|celery|frontend   (no && in start-all.bat, so no quoting problems)
call "%~dp0_config.bat"
if defined CONDA_ACTIVATE call "%CONDA_ACTIVATE%"
call "%~dp0start-%1.bat"
