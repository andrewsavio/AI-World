@echo off
rem Runs the AI World in the background with a tray icon (right-click it: Open dashboard / Stop world).
cd /d "%~dp0"
ollama show aiworld-gemma >nul 2>&1 || call :setup
start "" pythonw launcher.py
goto :eof

:setup
if exist models\gemma2-2b.gguf (
  pushd models & ollama create aiworld-gemma -f Modelfile & popd
) else (
  echo First run: downloading Gemma 2 (2B, about 1.6 GB) through Ollama...
  ollama pull gemma2:2b & ollama cp gemma2:2b aiworld-gemma
)
