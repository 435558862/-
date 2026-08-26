$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $ProjectRoot

Write-Host '检查 Python 3.11...'
$Python = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $Python) {
    throw '未找到 Python 3.11（64位）。请从 python.org 安装，并勾选 Add Python to PATH。'
}

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    & py -3.11 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
& .\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt

$Mpl = Join-Path $ProjectRoot 'storage\matplotlib'
New-Item -ItemType Directory -Force -Path $Mpl | Out-Null
$env:MPLCONFIGDIR = $Mpl
Write-Host '执行发布健康检查...'
& .\.venv\Scripts\python.exe scripts\health_check.py
if ($LASTEXITCODE -ne 0) { throw '发布健康检查失败，请查看上方问题后重试。' }

$Desktop = [Environment]::GetFolderPath('Desktop')
$Shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $Desktop 'ProphitBet 2.1.0.lnk'))
$Shortcut.TargetPath = (Join-Path $ProjectRoot 'app.bat')
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Save()
Write-Host '安装和健康检查完成。请双击桌面快捷方式或 app.bat 启动。' -ForegroundColor Green
