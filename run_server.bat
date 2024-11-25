@echo off
REM ANPR Server Wrapper Script
REM This script runs the ANPR API server and handles restart requests.
REM When the server exits and a .restart_flag file exists, it restarts.

setlocal enabledelayedexpansion

set "RESTART_FLAG=.restart_flag"
set "HOST=localhost"
set "PORT=8000"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :start_loop
if /i "%~1"=="--host" (
    set "HOST=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--port" (
    set "PORT=%~2"
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args

:start_loop
echo.
echo ========================================
echo ANPR Server Starting...
echo Host: %HOST%
echo Port: %PORT%
echo ========================================
echo.

REM Delete any existing restart flag
if exist "%RESTART_FLAG%" del "%RESTART_FLAG%"

REM Run the server
python src/main.py --mode api --host %HOST% --port %PORT%

REM Check if restart was requested
if exist "%RESTART_FLAG%" (
    echo.
    echo ========================================
    echo Restart requested. Restarting server...
    echo ========================================
    del "%RESTART_FLAG%"
    timeout /t 2 /nobreak >nul
    goto :start_loop
)

echo.
echo Server stopped.
pause
