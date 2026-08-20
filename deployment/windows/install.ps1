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
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'

$Desktop = [Environment]::GetFolderPath('Desktop')
$Shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $Desktop '足球预测系统.lnk'))
$Shortcut.TargetPath = (Join-Path $ProjectRoot 'app.bat')
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Save()
Write-Host '安装及测试完成。请双击项目目录中的 app.bat 启动。' -ForegroundColor Green
