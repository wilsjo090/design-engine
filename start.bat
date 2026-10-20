@echo off
rem Double-click to start the Design Engine editor. It opens in your browser.
rem Options are passed on, e.g.:  start.bat --port 8766   or   start.bat --no-browser
setlocal
title Design Engine
cd /d "%~dp0"

set "PY=%USERPROFILE%\miniconda3\envs\synkad\python.exe"
if not exist "%PY%" (
    echo The Python environment was not found at:
    echo   %PY%
    echo.
    echo Create it once with:
    echo   conda create -n synkad python=3.12
    echo   %PY% -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

"%PY%" app.py %*
set "CODE=%ERRORLEVEL%"

rem Keep the window open if something went wrong, so the message can be read.
if not "%CODE%"=="0" (
    echo.
    echo Design Engine stopped with an error.
    pause
)
exit /b %CODE%
