@echo off
rem Double-click to start the Design Engine editor. It opens in your browser.
rem Options are passed on, e.g.:  start.bat --port 8766   or   start.bat --no-browser
rem
rem Which Python: DESIGN_ENGINE_PYTHON if set (also from local.bat next to this file, not in the
rem repo), else a conda environment named design-engine, else the active conda environment,
rem else python on the PATH.
setlocal
title Design Engine
cd /d "%~dp0"

if exist "%~dp0local.bat" call "%~dp0local.bat"

set "PY="
if defined DESIGN_ENGINE_PYTHON set "PY=%DESIGN_ENGINE_PYTHON%"
if not defined PY for %%R in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\miniconda3") do (
    if not defined PY if exist "%%~R\envs\design-engine\python.exe" set "PY=%%~R\envs\design-engine\python.exe"
)
if not defined PY if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PY=%CONDA_PREFIX%\python.exe"
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"

if not defined PY goto nopython
if not exist "%PY%" goto nopython

"%PY%" app.py %*
set "CODE=%ERRORLEVEL%"

rem Keep the window open if something went wrong, so the message can be read.
if not "%CODE%"=="0" (
    echo.
    echo Design Engine stopped with an error.
    pause
)
exit /b %CODE%

:nopython
echo No Python environment for the Design Engine was found.
echo.
echo Create one once with:
echo   conda create -n design-engine python=3.12
echo   conda activate design-engine
echo   pip install -r requirements.txt
echo.
echo Or point to an existing one in local.bat next to start.bat:
echo   set "DESIGN_ENGINE_PYTHON=C:\path\to\python.exe"
echo.
pause
exit /b 1
