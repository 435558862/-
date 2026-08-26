#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$APP_DIR/.venv/bin/python"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "此启动器只能在 macOS 上运行。" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "尚未安装运行环境，请先执行：$APP_DIR/install-macos.sh" >&2
  read -r -p "按回车键关闭…" _
  exit 1
fi

cd "$APP_DIR"
mkdir -p storage/logs storage/matplotlib
export MPLCONFIGDIR="$APP_DIR/storage/matplotlib"
export PYTHONUNBUFFERED=1

LOCK_DIR="${TMPDIR:-/tmp}/com.prophitbet.app.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  LOCK_PID="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ -n "$LOCK_PID" ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
    echo "ProphitBet 已经在运行。"
    exit 0
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
fi
echo $$ > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM

"$PYTHON" "$APP_DIR/app.py" "$@"
