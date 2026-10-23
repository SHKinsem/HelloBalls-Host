@echo off
rem HelloBalls web console: double-click to start, then use the browser page.
cd /d "%~dp0"
set "PY=%USERPROFILE%\.venvs\helloballs-host\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -m helloballs.ui.server %*
pause
