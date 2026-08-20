# 足球预测系统 Windows 安装与维护

系统要求：Windows 10/11 64位、Python 3.11 64位、至少 8 GB 内存和约 6 GB 可用空间。

首次安装：解压完整压缩包，安装 Python 3.11 并勾选 `Add Python to PATH`，然后双击 `install_windows.bat`。安装完成后双击 `app.bat`。

日常使用：在软件里点击“手动同步竞猜数据并预测”。9个联赛的45个模型会按联赛自动调用。

后续更新：先关闭软件，将新版文件覆盖到原目录，然后双击 `update_windows.bat`。更新程序会先备份联赛数据、模型和竞猜记录，再更新依赖并运行测试。备份保存在 `windows-update-backups`。

不要删除：`storage/leagues` 包含45个模型及训练数据；`storage/network` 包含联赛配置和同步历史。

常见问题：若启动后立即关闭，打开命令提示符，进入软件目录运行 `.venv\Scripts\python.exe app.py` 查看错误。若浏览器同步失败，请更新 Chrome 或 Edge；Selenium 会使用浏览器驱动管理功能。
