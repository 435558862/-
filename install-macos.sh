#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "此脚本只能在 macOS 上运行。" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "未找到 Python 3。请先安装 Python 3.11，然后重新运行。" >&2
  exit 1
fi

PY_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VERSION" != "3.11" ]]; then
  echo "当前 Python 为 $PY_VERSION；本版本要求 Python 3.11。" >&2
  exit 1
fi

cd "$APP_DIR"
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements-macos.txt

if [[ "$(uname -m)" == "arm64" ]]; then
  .venv/bin/pip install tensorflow-macos==2.15.0
else
  .venv/bin/pip install tensorflow==2.15.1
fi

mkdir -p storage/logs storage/matplotlib storage/jingcai
.venv/bin/python scripts/health_check.py

echo
echo "安装完成。双击 start-macos.command，或运行："
echo "  $APP_DIR/start-macos.command"
