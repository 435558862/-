#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$APP_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "尚未安装运行环境，请先执行：$APP_DIR/install-wsl.sh" >&2
  exit 1
fi

cd "$APP_DIR"
mkdir -p "$APP_DIR/storage/matplotlib"

export MPLCONFIGDIR="$APP_DIR/storage/matplotlib"
export PYTHONUNBUFFERED=1
export QT_X11_NO_MITSHM=1

LOCK_FILE="/tmp/prophitbet-${UID}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "ProphitBet 已经在运行。"
  exit 0
fi

# WSLg 同时提供 X11 和 Wayland；PyQt 官方 wheel 的 xcb 后端兼容性更稳定。
if [[ -n "${DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi

exec "$PYTHON" "$APP_DIR/app.py" "$@"
