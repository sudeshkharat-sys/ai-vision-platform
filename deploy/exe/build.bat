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
echo [1/6] Checking Python...
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
echo [2/6] Checking Node.js / npm...
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
echo [3/6] Building React frontend...
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
echo [4/6] Installing Python build dependencies...
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
echo [5/6] Checking portable service binaries...

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
echo [6/6] Running PyInstaller...
cd /d "%DEPLOY_DIR%"

:: Build on D: drive to avoid C: space issues and Defender file locks.
:: Output goes straight to D:\AIVision-App — no xcopy needed.
set "PYI_TEMP=D:\aivision-build-temp"
set "PYI_WORK=%PYI_TEMP%\work"
set "PYI_DIST=D:\AIVision-App"

echo      PyInstaller temp : %PYI_WORK%
echo      Final output     : %PYI_DIST%\AIVision\aivision.exe

python -m PyInstaller launcher.spec --noconfirm ^
    --workpath "%PYI_WORK%" ^
    --distpath "%PYI_DIST%"
if errorlevel 1 ( echo [ERROR] PyInstaller failed. & exit /b 1 )

:: --------------------------------------------------------------------------
:: Done
:: --------------------------------------------------------------------------
echo.
echo ============================================================
echo  BUILD COMPLETE
echo  Output: D:\AIVision-App\AIVision\aivision.exe
echo ============================================================
echo.
echo To run:
echo   cd D:\AIVision-App\AIVision
echo   aivision.exe
echo.
echo On first launch, the app will:
echo   - Initialize the PostgreSQL database
echo   - Start all services
echo   - Open your browser at http://localhost:8000

endlocal
