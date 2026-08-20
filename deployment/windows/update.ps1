$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $ProjectRoot
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$BackupRoot = Join-Path $ProjectRoot "windows-update-backups\$Stamp"
New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
Copy-Item 'storage\leagues' $BackupRoot -Recurse -Force
Copy-Item 'storage\jingcai' $BackupRoot -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "模型和竞猜数据已备份到 $BackupRoot"
& (Join-Path $PSScriptRoot 'install.ps1')
Write-Host '更新维护完成。若新版异常，可从上述备份目录恢复 storage。' -ForegroundColor Green
