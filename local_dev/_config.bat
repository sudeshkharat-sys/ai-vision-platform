@echo off
rem ===== Per-project settings. Copy this folder to the other project and edit only this block. =====
set "PROJECT_NAME=ai-vision"
rem Short prefix for the typed commands, e.g. fa-backend. Use another prefix (vi) for the other project.
set "PROJECT_SHORT=vi"
set "PROJ=D:\Codes\Dataeaze\ai-vision-platform"
set "CONDA_ENV=ai-vision"
set "BACKEND_APP=app.main:app"
set "BACKEND_PORT=8000"
set "CELERY_APP=app.tasks.celery_app"
rem Leave FRONTEND_PORT empty to use the project's own default.
set "FRONTEND_PORT="
rem ===== Shared tools (same for every project) =====
set "PGROOT=C:\postgresql-17.8-1-windows-x64-binaries\pgsql"
set "PGBIN=%PGROOT%\bin"
set "PGDATA=%PGROOT%\data"
set "PGLOG=C:\postgresql-17.8-1-windows-x64-binaries\pg.log"
set "REDISDIR=C:\redis-latest"
rem Miniconda root differs per machine. First one that exists wins.
set "CONDA_ACTIVATE="
if exist "C:\conda\Scripts\activate.bat" set "CONDA_ACTIVATE=C:\conda\Scripts\activate.bat"
if not defined CONDA_ACTIVATE if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if not defined CONDA_ACTIVATE if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat"
