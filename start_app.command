#!/bin/zsh
# Double-click this file in Finder, or run: ./start_app.command
set -e

export KMP_DUPLICATE_LIB_OK=TRUE
cd "$(dirname "$0")"
VENV_DIR=".venv"
PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[1/3] Creating Python virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

echo "[2/3] Installing or updating app dependencies..."
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt

if command -v ollama >/dev/null 2>&1 && ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama in the background..."
  ollama serve >/tmp/ttal-ollama.log 2>&1 &
fi

echo "[3/3] Starting TTAL at http://127.0.0.1:8000"
echo "Keep this Terminal window open while using the app."
(sleep 2; open http://127.0.0.1:8000) &
exec "$PYTHON_BIN" app.py
