@echo off
rem Starts the AI World: sets up Gemma in Ollama (first time only), serves the world on http://localhost:8000 and opens it.
cd /d "%~dp0"
ollama show aiworld-gemma >nul 2>&1 || call :setup
python server.py
goto :eof

:setup
if exist models\gemma2-2b.gguf (
  pushd models & ollama create aiworld-gemma -f Modelfile & popd
) else (
  echo First run: downloading Gemma 2 (2B, about 1.6 GB) through Ollama...
  ollama pull gemma2:2b & ollama cp gemma2:2b aiworld-gemma
)
