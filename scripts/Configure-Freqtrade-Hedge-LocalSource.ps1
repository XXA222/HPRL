[CmdletBinding()]
param(
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
}
else {
    $ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SitePackages = Join-Path $ProjectRoot ".venv\Lib\site-packages"
$ClientRoot = Join-Path $ProjectRoot "ft_client"
$PthPath = Join-Path $SitePackages "_freqtrade_hedge_local_source.pth"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw ("Project-local Python not found: " + $Python)
}
if (-not (Test-Path -LiteralPath $SitePackages -PathType Container)) {
    throw ("Project-local site-packages not found: " + $SitePackages)
}
if (-not (Test-Path -LiteralPath (Join-Path $ClientRoot "freqtrade_client") -PathType Container)) {
    throw ("Local freqtrade_client source not found: " + $ClientRoot)
}

$Encoding = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines(
    $PthPath,
    @($ProjectRoot, $ClientRoot),
    $Encoding
)

$ProbeRoot = Split-Path -Parent $ProjectRoot
Push-Location -LiteralPath $ProbeRoot
try {
    & $Python -c "import pathlib,sys,freqtrade,freqtrade_client; root=pathlib.Path(r'$ProjectRoot').resolve(); client=(root/'ft_client').resolve(); fm=pathlib.Path(freqtrade.__file__).resolve(); cm=pathlib.Path(freqtrade_client.__file__).resolve(); print('Python:',sys.executable); print('Freqtrade:',fm); print('FreqtradeClient:',cm); print('PTH:',r'$PthPath'); assert root in fm.parents, (root,fm); assert client in cm.parents, (client,cm)"
    if ($LASTEXITCODE -ne 0) {
        throw "Local source registration verification failed."
    }
}
finally {
    Pop-Location
}

Write-Host ("Local source registration: " + $PthPath) -ForegroundColor Green
Write-Host "No editable install was used." -ForegroundColor Green
