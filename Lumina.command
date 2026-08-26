#!/bin/bash
# Double-click launcher for Lumina (macOS).
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Creating virtualenv and installing dependencies (one time)…"
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/python main.py "$@"
