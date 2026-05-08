@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  AI Vision Platform - Windows EXE Build Script
echo ============================================================

:: Resolve repo root (two levels up from deploy\exe)
set "DEPLOY_DIR=%~dp0"
pushd "%~dp0..\.."
set "REPO_ROOT=%CD%"
popd

echo [DEBUG] DEPLOY_DIR = %DEPLOY_DIR%
echo [DEBUG] REPO_ROOT  = %REPO_ROOT%

:: --------------------------------------------------------------------------
:: Step 1 - Check Python
:: --------------------------------------------------------------------------
echo.
echo [1/8] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11 from https://www.python.org/downloads/
    exit /b 1
)
python --version

:: --------------------------------------------------------------------------
:: Step 2 - Check Node
:: --------------------------------------------------------------------------
echo.
echo [2/8] Checking Node.js / npm...
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org/
    exit /b 1
)
node --version
call npm --version

:: --------------------------------------------------------------------------
:: Step 3 - Build React frontend
:: --------------------------------------------------------------------------
echo.
echo [3/8] Building React frontend...
cd /d "%REPO_ROOT%\frontend"

if not exist node_modules (
    echo      Installing npm packages...
    call npm install --legacy-peer-deps
    if errorlevel 1 ( echo [ERROR] npm install failed. & exit /b 1 )
)

set REACT_APP_API_URL=http://localhost:8000/api/v1
set REACT_APP_BASE_URL=http://localhost:8000
call npm run build
if errorlevel 1 ( echo [ERROR] React build failed. & exit /b 1 )
echo      Frontend built successfully.

:: --------------------------------------------------------------------------
:: Step 4 - Install Python build dependencies
:: --------------------------------------------------------------------------
echo.
echo [4/8] Installing Python build dependencies...
cd /d "%REPO_ROOT%"
pip install pyinstaller ^
    fastapi uvicorn[standard] sqlalchemy[asyncio] asyncpg psycopg2-binary ^
    celery[redis] redis transformers sentencepiece protobuf ultralytics ^
    Pillow supervision opencv-python-headless pyyaml python-dotenv ^
    pydantic pydantic-settings passlib[bcrypt] python-jose[cryptography] ^
    python-multipart websockets loguru
if errorlevel 1 ( echo [ERROR] pip install failed. & exit /b 1 )

:: --------------------------------------------------------------------------
:: Step 5 - Check for portable binaries
:: --------------------------------------------------------------------------
echo.
echo [5/8] Checking portable service binaries...

:: Always cd back to DEPLOY_DIR before file checks so relative logic is correct
cd /d "%DEPLOY_DIR%"

if exist "resources\postgres\bin\pg_ctl.exe" (
    echo      PostgreSQL binaries found.
) else (
    echo.
    echo [WARNING] Portable PostgreSQL NOT found.
    echo          Expected: %DEPLOY_DIR%resources\postgres\bin\pg_ctl.exe
    echo.
    echo   Download the binary ZIP from:
    echo   https://www.enterprisedb.com/download-postgresql-binaries
    echo   Extract bin\ lib\ share\ into: deploy\exe\resources\postgres\
    echo.
    set /p CONTINUE="Continue build without PostgreSQL binaries? [y/N]: "
    if /i "!CONTINUE!" neq "y" exit /b 1
)

if exist "resources\redis\redis-server.exe" (
    echo      Redis binary found.
) else (
    echo.
    echo [WARNING] Portable Redis NOT found.
    echo          Expected: %DEPLOY_DIR%resources\redis\redis-server.exe
    echo.
    echo   Download from: https://github.com/microsoftarchive/redis/releases
    echo   Place redis-server.exe and redis-cli.exe in: deploy\exe\resources\redis\
    echo.
    set /p CONTINUE="Continue build without Redis binary? [y/N]: "
    if /i "!CONTINUE!" neq "y" exit /b 1
)

:: --------------------------------------------------------------------------
:: Step 6 - Run PyInstaller
:: --------------------------------------------------------------------------
echo.
echo [6/8] Running PyInstaller...
cd /d "%DEPLOY_DIR%"

:: Build on D: drive to avoid C: space issues and Defender file locks.
:: Output goes straight to D:\AIVision-App — no xcopy needed.
set "PYI_TEMP=D:\aivision-build-temp"
set "PYI_WORK=%PYI_TEMP%\work"
set "PYI_DIST=D:\AIVision-App"

echo      PyInstaller temp : %PYI_WORK%
echo      Final output     : %PYI_DIST%\AIVision\aivision.exe

:: Kill any running aivision.exe so PyInstaller can overwrite it.
:: taskkill returns 128 when the process is not found — that is fine.
echo      Stopping any running aivision.exe...
taskkill /f /im aivision.exe >nul 2>&1

:: Give Windows a moment to fully release file handles after the kill.
timeout /t 3 /nobreak >nul

:: Manually remove the old dist folder so PyInstaller never has to rmtree
:: a directory that Windows Defender or AV may still have open.
if exist "%PYI_DIST%\AIVision" (
    echo      Removing previous build at %PYI_DIST%\AIVision...
    rd /s /q "%PYI_DIST%\AIVision"
    if errorlevel 1 (
        echo [ERROR] Could not remove %PYI_DIST%\AIVision — close any open
        echo         files or Explorer windows inside that folder and retry.
        exit /b 1
    )
)

:: Wipe the PyInstaller work/cache directory so no stale model files (.pt,
:: images, etc.) from a previous build are re-bundled into the new EXE.
if exist "%PYI_WORK%" (
    echo      Clearing PyInstaller cache at %PYI_WORK%...
    rd /s /q "%PYI_WORK%"
)

python -m PyInstaller launcher.spec --noconfirm ^
    --workpath "%PYI_WORK%" ^
    --distpath "%PYI_DIST%"
if errorlevel 1 ( echo [ERROR] PyInstaller failed. & exit /b 1 )

:: --------------------------------------------------------------------------
:: Step 7 — Pre-download YOLO base weights into the dist folder
::
:: These weights land in data\yolo_weights\ inside the dist folder and ARE
:: included in the distribution ZIP (package.bat only excludes trained project
:: models under data\models\ and .png files).
:: This makes the deployment fully offline — no internet needed on end-user
:: machines.
:: --------------------------------------------------------------------------
echo.
echo [7/8] Downloading YOLO base weights (included in ZIP for offline use)...
echo       Target  : %PYI_DIST%\AIVision\data\yolo_weights
echo       ~600 MB - 1.5 GB depending on which model families are available.
echo.

set "YOLO_WEIGHTS_DIR=%PYI_DIST%\AIVision\data\yolo_weights"
if not exist "%YOLO_WEIGHTS_DIR%" mkdir "%YOLO_WEIGHTS_DIR%"

:: Run the download script from the repo root so the import paths are correct
cd /d "%REPO_ROOT%"
python backend\scripts\download_yolo_weights.py
if errorlevel 1 (
    echo [WARNING] Some YOLO weights could not be downloaded.
    echo          The launcher will retry at first launch (requires internet).
) else (
    echo      All available YOLO weights downloaded to local dist folder.
)

:: --------------------------------------------------------------------------
:: Step 8 — Package into distributable ZIP
:: --------------------------------------------------------------------------
echo.
echo [8/8] Creating distribution ZIP...
call "%DEPLOY_DIR%package.bat" "%PYI_DIST%\AIVision" "%PYI_DIST%"
if errorlevel 1 (
    echo [WARNING] ZIP packaging failed, but the EXE build succeeded.
    echo           Run package.bat manually to create the ZIP.
)

:: --------------------------------------------------------------------------
:: Done
:: --------------------------------------------------------------------------
echo.
echo ============================================================
echo  BUILD COMPLETE
echo  EXE     : %PYI_DIST%\AIVision\aivision.exe
echo  Weights : %PYI_DIST%\AIVision\data\yolo_weights\
echo  ZIP     : %PYI_DIST%\AIVision-win64-*.zip
echo ============================================================
echo.
echo To test locally:
echo   cd %PYI_DIST%\AIVision
echo   aivision.exe
echo.
echo To distribute: send the AIVision-win64-*.zip file.
echo On first launch, the app will:
echo   - Initialize the PostgreSQL database
echo   - Start all services
echo   - Open the browser at http://localhost:8000

endlocal
