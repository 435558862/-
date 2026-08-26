#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$APP_DIR/.venv/bin/python"
LABEL="com.prophitbet.daily-sporttery"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "此脚本只能在 macOS 上运行。" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "请先运行 $APP_DIR/install-macos.sh" >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$APP_DIR/storage/jingcai"
SAFE_APP_DIR="${APP_DIR//&/&amp;}"

sed "s|__APP_DIR__|$SAFE_APP_DIR|g" \
  "$APP_DIR/deployment/macos/com.prophitbet.daily-sporttery.plist.in" > "$PLIST"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "每日同步任务已安装：$PLIST"
