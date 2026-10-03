@echo off
rem Runs the AI World in the background with a tray icon (right-click it: Open dashboard / Stop world).
cd /d "%~dp0"
start "" pythonw launcher.py
