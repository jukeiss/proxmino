#!/usr/bin/env bash
# Sets up the virtual environment on first run, then starts the web app.
set -euo pipefail
CALLER_DIR="$PWD"
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 /opt/homebrew/bin/python3.12 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "Python 3.10 or newer is required."; exit 1; fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

if [ "${1:-}" = "test" ]; then exec .venv/bin/python -m pytest -q; fi
if [ "${1:-}" = "ask" ]; then shift; exec .venv/bin/python -m proxmino ask "$@"; fi
if [ "${1:-}" = "code" ]; then shift; exec .venv/bin/python -m proxmino code --dir "$CALLER_DIR" "$@"; fi
exec .venv/bin/python -m proxmino serve "$@"
