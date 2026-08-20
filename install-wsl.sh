#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${HOME}/.local/bin/uv"

cd "$APP_DIR"

echo "[1/4] 准备 uv 和 Python 3.11..."
if [[ ! -x "$UV_BIN" ]]; then
  python3 -m pip install --user --break-system-packages uv
fi
"$UV_BIN" python install 3.11

echo "[2/4] 创建项目虚拟环境..."
if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
  "$UV_BIN" venv --python 3.11 "$APP_DIR/.venv"
fi

echo "[3/4] 安装项目依赖..."
"$UV_BIN" pip install \
  --default-index "https://pypi.tuna.tsinghua.edu.cn/simple" \
  --python "$APP_DIR/.venv/bin/python" \
  --requirement "$APP_DIR/requirements-wsl.txt"

echo "[4/4] 运行测试..."
mkdir -p "$APP_DIR/storage/matplotlib"
MPLCONFIGDIR="$APP_DIR/storage/matplotlib" \
  QT_QPA_PLATFORM=offscreen \
  "$APP_DIR/.venv/bin/python" -m pytest -q

echo "安装完成。运行 ./start.sh 启动 ProphitBet。"
