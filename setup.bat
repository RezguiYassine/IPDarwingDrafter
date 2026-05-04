@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM IP DrawingDrafter - AP3 Vectorization Pipeline · Setup Script (Windows)
REM
REM Run this once after cloning the repo, from a Command Prompt or PowerShell:
REM
REM     setup.bat
REM
REM It will:
REM   1. Create a Python virtual environment in .venv\
REM   2. Install runtime dependencies from requirements.txt
REM   3. Download sketchcleannet.pth (or print manual instructions)
REM
REM Idempotent - safe to re-run. Linux/macOS users should run setup.sh instead.
REM ─────────────────────────────────────────────────────────────────────────────

setlocal EnableDelayedExpansion

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

set "PYTHON=py -3"
set "VENV_DIR=.venv"

REM ── 1. Python virtual environment ────────────────────────────────────────────
if exist "%VENV_DIR%\Scripts\python.exe" goto :venv_ready

echo ^> Creating virtualenv in %VENV_DIR%\ ...
%PYTHON% -m venv "%VENV_DIR%"
if errorlevel 1 goto :err_no_python
goto :venv_done

:venv_ready
echo ^> Virtualenv %VENV_DIR%\ already exists, skipping creation.

:venv_done
call "%VENV_DIR%\Scripts\activate.bat"

REM ── 2. Install dependencies ──────────────────────────────────────────────────
echo ^> Upgrading pip ...
python -m pip install --upgrade pip --quiet

echo ^> Installing requirements.txt ...
python -m pip install -r requirements.txt
if errorlevel 1 goto :err_pip

REM ── 3. Model weights ─────────────────────────────────────────────────────────
set "SKETCHCLEAN_WEIGHT=%PROJECT_ROOT%models\sketchcleannet.pth"

REM OneDrive (HAW Landshut) folder containing all model weights.
set "ONEDRIVE_FOLDER_URL=https://hawlandshutde-my.sharepoint.com/:f:/g/personal/rey21950_az_haw-landshut_de/IgBt1cX39WruQ4Oxg9neSeQbAXvmPFL6QjFBUjJYiiT4k_M?e=eCbhJK"

REM Per-file direct download URL for sketchcleannet.pth.
set "DIRECT_DOWNLOAD_URL=https://hawlandshutde-my.sharepoint.com/:u:/g/personal/rey21950_az_haw-landshut_de/IQDKTxEjVJbnQLn6ZLOkpf2oAf0q9wUvBD1gfhYst_nS2T8?e=PlKpSA"

echo.
echo -- Model weights --

if exist "%SKETCHCLEAN_WEIGHT%" goto :weight_present
if not "%DIRECT_DOWNLOAD_URL%"=="" goto :weight_download
goto :weight_manual

:weight_present
echo [OK] sketchcleannet.pth already present at %SKETCHCLEAN_WEIGHT%
goto :weights_done

:weight_download
echo ^> Downloading sketchcleannet.pth ...
where curl >nul 2>&1
if errorlevel 1 goto :weight_download_ps
curl -L -o "%SKETCHCLEAN_WEIGHT%" "%DIRECT_DOWNLOAD_URL%"
if errorlevel 1 goto :weight_manual
echo [OK] Saved to %SKETCHCLEAN_WEIGHT%
goto :weights_done

:weight_download_ps
echo    curl not found, falling back to PowerShell ...
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%DIRECT_DOWNLOAD_URL%' -OutFile '%SKETCHCLEAN_WEIGHT%'"
if errorlevel 1 goto :weight_manual
echo [OK] Saved to %SKETCHCLEAN_WEIGHT%
goto :weights_done

:weight_manual
echo.
echo [!] sketchcleannet.pth is NOT YET INSTALLED.
echo.
echo   Option A - Download manually:
echo     1. Open this OneDrive folder in your browser:
echo          %ONEDRIVE_FOLDER_URL%
echo     2. Click 'sketchcleannet.pth' and download it.
echo     3. Move the downloaded file to:
echo          %SKETCHCLEAN_WEIGHT%
echo.
echo   Option B - Skip it.
echo     Stage 1 will fall back to its classical cleaning mode.
echo     The pipeline still produces valid output; just lower quality on
echo     photographed/shaded sketches.
echo.
echo The other two model files ^(puhachov_keypoints.pth, free2cad_v3_best.pth^)
echo are small enough to ship in the repository and are already in place.

:weights_done
echo.
echo -- Setup complete --
echo Activate the environment with:
echo     %VENV_DIR%\Scripts\activate.bat        ^(Command Prompt^)
echo     %VENV_DIR%\Scripts\Activate.ps1        ^(PowerShell^)
echo Then run the pipeline. See README.md for the full command.
goto :eof

:err_no_python
echo.
echo ERROR: failed to create virtualenv.
echo Make sure Python 3.10+ is installed and the "py" launcher is on your PATH.
echo Download from https://www.python.org/downloads/windows/
echo During installation, check "Add Python to PATH".
exit /b 1

:err_pip
echo.
echo ERROR: pip install failed. Check your internet connection and re-run.
exit /b 1
