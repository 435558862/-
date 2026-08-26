$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$DistRoot = Join-Path $ProjectRoot 'dist'
$Output = Join-Path $DistRoot 'ProphitBet-Windows'
$Zip = Join-Path $DistRoot 'ProphitBet-Windows.zip'

function Copy-CleanTree([string]$Source, [string]$Destination) {
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    & robocopy $Source $Destination /E /NFL /NDL /NJH /NJS /NP `
        /XD __pycache__ .pytest_cache backups logs matplotlib raw reports lineups audits `
        /XF *.pyc *.orig *.rej *.bak-* .api_football_key sync_state.json browser.json
    if ($LASTEXITCODE -gt 7) { throw "复制失败：$Source（robocopy=$LASTEXITCODE）" }
}

if (Test-Path $Output) { Remove-Item $Output -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$RootFiles = @(
    'app.py', 'app.bat', 'install_windows.bat', 'update_windows.bat',
    'requirements-windows.txt', 'README.md', 'README-Windows.md',
    'CHANGELOG.md', 'LICENSE.txt', 'THIRD_PARTY_NOTICES.md'
)
foreach ($Item in $RootFiles) { Copy-Item (Join-Path $ProjectRoot $Item) $Output -Force }
Copy-CleanTree (Join-Path $ProjectRoot 'src') (Join-Path $Output 'src')
Copy-CleanTree (Join-Path $ProjectRoot 'scripts') (Join-Path $Output 'scripts')
Copy-CleanTree (Join-Path $ProjectRoot 'deployment\windows') (Join-Path $Output 'deployment\windows')
Copy-CleanTree (Join-Path $ProjectRoot 'storage\leagues') (Join-Path $Output 'storage\leagues')
Copy-CleanTree (Join-Path $ProjectRoot 'storage\graphics') (Join-Path $Output 'storage\graphics')
Copy-CleanTree (Join-Path $ProjectRoot 'storage\models') (Join-Path $Output 'storage\models')

$Network = Join-Path $Output 'storage\network'
New-Item -ItemType Directory -Force -Path $Network | Out-Null
foreach ($Name in @('leagues.json','models.json','sporttery_team_aliases.json',
                     'k_league_oddsportal_2014_2021.csv','k_league_sgodds.csv')) {
    $Source = Join-Path $ProjectRoot "storage\network\$Name"
    if (Test-Path $Source) { Copy-Item $Source $Network -Force }
}
$Learning = Join-Path $Output 'storage\jingcai\learning'
New-Item -ItemType Directory -Force -Path $Learning | Out-Null
foreach ($Name in @('official_market_history.csv','selection_profile.json',
                     'settled_predictions.csv','status.json')) {
    $Source = Join-Path $ProjectRoot "storage\jingcai\learning\$Name"
    if (Test-Path $Source) { Copy-Item $Source $Learning -Force }
}
foreach ($Directory in @('storage\jingcai\raw','storage\jingcai\reports',
                          'storage\jingcai\lineups','storage\logs','storage\matplotlib')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Output $Directory) | Out-Null
}
if (Test-Path $Zip) { Remove-Item $Zip -Force }
Compress-Archive -Path "$Output\*" -DestinationPath $Zip -CompressionLevel Optimal
$Hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
"$Hash  ProphitBet-Windows.zip" | Set-Content `
    (Join-Path $DistRoot 'ProphitBet-Windows.sha256') -Encoding ascii
Write-Host "Windows发布包已生成：$Zip" -ForegroundColor Green
Write-Host "SHA256：$Hash" -ForegroundColor Green
