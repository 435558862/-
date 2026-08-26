# ProphitBet 2.1.0 — macOS 部署

macOS 与 Windows/WSL 使用同一套 `app.py`、`src`、模型格式和存储结构；仅安装、启动脚本不同。

## 环境

- macOS 12 或更高版本
- Python 3.11（不要使用 3.12/3.13）
- 首次安装需要网络
- Apple Silicon（M1/M2/M3/M4）和 Intel Mac 均支持

## 首次安装

1. 将完整项目复制到 Mac，例如 `~/Applications/ProphitBet`。
2. 打开“终端”，执行：

   ```bash
   cd ~/Applications/ProphitBet
   chmod +x install-macos.sh install-macos.command start-macos.command update-macos.sh
   ./install-macos.sh
   ```

3. 安装完成后，可双击 `start-macos.command` 启动。

若系统拦截首次运行，在“系统设置 → 隐私与安全性”中允许本次启动。不要关闭系统安全保护。

## 保持两端代码一致

- Windows/WSL 与 Mac 都从同一个 Git 分支更新。
- 发布版本使用相同版本号和提交号；Mac 上运行 `git rev-parse HEAD` 可核对。
- `storage` 是模型与用户数据，不应通过普通代码更新覆盖。
- 更新前备份 `storage`；Git 部署可运行 `./update-macos.sh`。

## 迁移现有模型数据

首次部署时，在两边程序均已退出的情况下，将 WSL 项目的 `storage` 整体复制到 Mac 项目根目录。不要只复制单个 `.pkl`，模型索引、球队映射、校准参数和历史特征需要一起迁移。

模型数据不建议让两台机器同时训练后再互相覆盖。日常应指定一台为“训练主机”，另一台只同步训练完成的完整 `storage` 备份。

## 诊断

```bash
cd ~/Applications/ProphitBet
.venv/bin/python scripts/health_check.py
tail -n 100 storage/logs/app.log
```

程序不承诺固定命中率或收益；发布数据应采用严格时间切分的独立测试结果。

## 可选：关机后补采与定时同步

macOS 登录期间可安装定时同步：

```bash
chmod +x deployment/macos/*.sh
./deployment/macos/install_daily_sync.sh
```

电脑关机期间系统无法联网采集；再次登录后任务会恢复。移除任务运行 `./deployment/macos/uninstall_daily_sync.sh`。
