@echo off
call "%~dp0_config.bat"
call "%CONDA_ACTIVATE%" %CONDA_ENV%
cd /d "%PROJ%"
python -m pip install -r backend\requirements.txt
