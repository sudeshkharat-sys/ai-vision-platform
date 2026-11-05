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
rem Several conda installs can exist. Use the one that actually contains the env named in CONDA_ENV.
set "CONDA_ACTIVATE="
for %%R in ("C:\conda" "%USERPROFILE%\AppData\Local\miniconda3_new" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3") do (
  if not defined CONDA_ACTIVATE if exist "%%~R\envs\%CONDA_ENV%\python.exe" if exist "%%~R\Scripts\activate.bat" set "CONDA_ACTIVATE=%%~R\Scripts\activate.bat"
)
if not defined CONDA_ACTIVATE (
  echo.
  echo [ERROR] No conda env named "%CONDA_ENV%" found in any known conda install. Create it first.
)
