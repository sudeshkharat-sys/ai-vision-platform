@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  AI Vision Platform - Distribution Package Creator
echo ============================================================

:: Default paths — override via arguments:
::   package.bat [dist-folder] [output-folder]
::   e.g.  package.bat "D:\AIVision-App\AIVision" "C:\Users\me\Desktop"
set "DIST_DIR=D:\AIVision-App\AIVision"
set "OUTPUT_DIR=D:\AIVision-App"

if not "%~1"=="" set "DIST_DIR=%~1"
if not "%~2"=="" set "OUTPUT_DIR=%~2"

:: --------------------------------------------------------------------------
:: Validate that the build exists
:: --------------------------------------------------------------------------
if not exist "%DIST_DIR%\aivision.exe" (
    echo.
    echo [ERROR] aivision.exe not found at:
    echo         %DIST_DIR%\aivision.exe
    echo.
    echo         Run build.bat first to produce the executable.
    exit /b 1
)

:: --------------------------------------------------------------------------
:: Build a datestamped filename using PowerShell (avoids locale issues)
:: --------------------------------------------------------------------------
for /f "delims=" %%D in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "DATE_TAG=%%D"
set "ZIP_NAME=AIVision-win64-%DATE_TAG%.zip"
set "ZIP_PATH=%OUTPUT_DIR%\%ZIP_NAME%"

echo.
echo Source folder : %DIST_DIR%
echo Output ZIP    : %ZIP_PATH%
echo.
echo Included  : data\yolo_weights\*.pt  (pre-downloaded YOLO base weights — offline ready)
echo Excluded  : data\models\**\*.pt     (user-trained project models — runtime data)
echo Excluded  : **\*.png                (test images and previews)
echo.

:: Remove stale zip if present
if exist "%ZIP_PATH%" (
    echo [INFO] Removing previous %ZIP_NAME%...
    del /f "%ZIP_PATH%"
)

:: --------------------------------------------------------------------------
:: Step 1 — Create filtered ZIP
::
:: Compress-Archive has no exclusion support, so we use the .NET ZipFile
:: API directly in inline PowerShell.
::
:: Exclusion rules (path-aware, not just extension):
::
::   data\models\**\*.pt   — trained project weights produced at runtime.
::                           These are user data and change every training run;
::                           bundling them would include stale or wrong models.
::
::   **\*.png              — test images, annotation previews, screenshots.
::                           Not needed in the distributable bundle.
::
:: KEPT in ZIP:
::   data\yolo_weights\*.pt — pre-downloaded YOLO base weights (YOLOv8–v26,
::                            nano to XL).  Required for offline training —
::                            the deployment machine has NO internet access.
:: --------------------------------------------------------------------------
echo [1/2] Compressing AIVision folder...

powershell -NoProfile -Command ^
    "$src = '%DIST_DIR%'; $dst = '%ZIP_PATH%';" ^
    "$folderName = Split-Path $src -Leaf;" ^
    "Add-Type -AssemblyName 'System.IO.Compression.FileSystem';" ^
    "$mode  = [System.IO.Compression.ZipArchiveMode]::Create;" ^
    "$level = [System.IO.Compression.CompressionLevel]::Optimal;" ^
    "$zip   = [System.IO.Compression.ZipFile]::Open($dst, $mode);" ^
    "$skipped = 0; $count = 0;" ^
    "$files = Get-ChildItem -Path $src -Recurse -File;" ^
    "foreach ($f in $files) {" ^
    "    $rel = $f.FullName.Substring($src.Length + 1);" ^
    "    $ext = $f.Extension.ToLower();" ^
    "    $skip = $false;" ^
    "    if ($ext -eq '.png') { $skip = $true }" ^
    "    elseif ($ext -eq '.pt' -and $rel -like 'data\models\*') { $skip = $true }" ^
    "    if ($skip) { $skipped++; continue }" ^
    "    $entry = $folderName + '\\' + $rel;" ^
    "    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, $entry, $level) | Out-Null;" ^
    "    $count++;" ^
    "    if ($count %% 200 -eq 0) { Write-Host \"  ... $count files added ($skipped skipped)\" }" ^
    "};" ^
    "$zip.Dispose();" ^
    "Write-Host \"  Done: $count files packaged, $skipped files skipped\""

if errorlevel 1 (
    echo.
    echo [ERROR] ZIP creation failed.
    echo         Make sure PowerShell 5+ is available and the output path is writable.
    exit /b 1
)

:: --------------------------------------------------------------------------
:: Step 2 — Report size and instructions
:: --------------------------------------------------------------------------
echo [2/2] Verifying archive...

for %%A in ("%ZIP_PATH%") do (
    set "ZIP_BYTES=%%~zA"
)
set /a "ZIP_MB=%ZIP_BYTES% / 1048576"

echo.
echo ============================================================
echo  PACKAGE READY
echo.
echo  File     : %ZIP_PATH%
echo  Size     : ~%ZIP_MB% MB
echo  Included : data\yolo_weights\*.pt  (YOLO base weights — offline ready)
echo  Skipped  : data\models\**\*.pt and *.png
echo ============================================================
echo.
echo ---- Instructions for the end user --------------------------
echo.
echo  1. Send them: %ZIP_NAME%
echo.
echo  2. They should:
echo       a. Extract the ZIP to any folder
echo            e.g. right-click > Extract All... > C:\AIVision
echo.
echo       b. Open the extracted AIVision\ folder
echo.
echo       c. Double-click  aivision.exe
echo            (or right-click > Run as administrator if UAC blocks it)
echo.
echo       d. Wait ~30-60 seconds on first launch
echo            The app initialises the database automatically.
echo.
echo       e. A browser tab opens at http://localhost:8000
echo.
echo  NO installation required — no Python, no Node, no database setup.
echo  NO internet required — YOLO base weights are bundled in the ZIP.
echo  Everything needed for offline training is included.
echo.
echo  System requirements for the end user:
echo    - Windows 10/11 (64-bit)
echo    - NVIDIA GPU with CUDA 12.x drivers (for AI inference)
echo    - 8 GB RAM minimum (16 GB recommended)
echo    - 10 GB free disk space for the app + AI model weights
echo -------------------------------------------------------------

endlocal
