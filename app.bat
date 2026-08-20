@echo off
setlocal
cd /d "%~dp0"
set "MPLCONFIGDIR=%CD%\storage\matplotlib"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" app.py
) else (
  echo Windows环境尚未安装，请先双击 install_windows.bat
  pause
)
