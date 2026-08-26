#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

if [[ ! -d .git ]]; then
  echo "当前目录不是 Git 部署，不能自动更新。请使用新的完整发布包覆盖程序文件，并保留 storage 目录。" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "检测到本机代码有未提交修改，为避免覆盖，已停止更新。" >&2
  exit 1
fi

git pull --ff-only
./install-macos.sh
echo "代码和依赖已更新；storage 中的模型与历史数据保持不变。"
