#!/usr/bin/env bash
set -Eeuo pipefail
LABEL="com.prophitbet.daily-sporttery"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$PLIST"
echo "每日同步任务已移除；模型和历史数据未删除。"
