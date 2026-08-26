# ProphitBet 2.1.0

足球比赛数据分析与概率预测桌面软件，支持中国竞彩足球同步、联赛专用模型、市场基线、历史攻防双泊松模拟、赛果复盘和Excel导出。

## 重要说明

- 输出是概率分析，不构成收益保证或投注承诺。
- 模型准确率会随联赛、时间区间、数据完整性和推荐覆盖率变化。
- 首发、赔率和赛果依赖第三方数据源；软件会显示缺失或降级状态，不应把缺失数据当作确定结论。
- 发布或销售时必须同时附带 `LICENSE.txt`，保留上游MIT版权声明。

## 运行环境

- Windows 10/11 64位 + Python 3.11，或 Windows 11 WSL2/WSLg。
- 建议至少8GB内存、8GB可用磁盘空间。
- Windows安装与升级见 `README-Windows.md`。

## 启动

WSL：

```bash
./install-wsl.sh
./start.sh
```

Windows：双击 `install_windows.bat`，安装完成后双击 `app.bat`。

## 数据与隐私

模型、历史比赛、预测报告和日志默认仅保存在本机 `storage/`。阵容接口密钥保存在 `storage/network/.api_football_key` 或环境变量 `API_FOOTBALL_KEY`，该文件不得加入发布包或版本库。

## 故障诊断

运行 `python scripts/health_check.py`。应用日志位于 `storage/logs/app.log`，致命错误记录于 `storage/logs/fatal.log`。提交售后问题时请提供日志，但先检查其中是否含球队、时间或本地路径等不希望分享的信息。

## 上游与许可

本项目基于 Vasileios Kochliaridis 的 ProphitBet 开源项目二次开发。上游采用MIT许可证，允许商业使用、修改和销售，但必须保留版权与许可文本。

