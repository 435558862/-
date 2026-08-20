$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Output = Join-Path $ProjectRoot 'dist\ProphitBet-Windows'
if (Test-Path $Output) { Remove-Item $Output -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Output | Out-Null

$Items = @('app.py','app.bat','install_windows.bat','update_windows.bat','requirements-windows.txt','README-Windows.md','src','scripts','tests','deployment')
foreach ($Item in $Items) { Copy-Item (Join-Path $ProjectRoot $Item) $Output -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $Output 'storage') | Out-Null
foreach ($Item in @('leagues','network','graphics','reports')) {
    Copy-Item (Join-Path $ProjectRoot "storage\$Item") (Join-Path $Output 'storage') -Recurse -Force
}
Compress-Archive -Path "$Output\*" -DestinationPath (Join-Path $ProjectRoot 'dist\ProphitBet-Windows.zip') -Force
Write-Host 'Windows便携包已生成：dist\ProphitBet-Windows.zip' -ForegroundColor Green
