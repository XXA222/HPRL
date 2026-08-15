[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$FreqtradeArgs
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
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw ("Local virtual environment Python not found: " + $Python)
}

$LocalSourcePth = Join-Path $ProjectRoot ".venv\Lib\site-packages\_freqtrade_hedge_local_source.pth"
if (-not (Test-Path -LiteralPath $LocalSourcePth -PathType Leaf)) {
    throw (
        "Local source registration is missing. Run scripts\Configure-Freqtrade-Hedge-LocalSource.ps1 first."
    )
}

if ($null -eq $FreqtradeArgs -or $FreqtradeArgs.Count -eq 0) {
    $FreqtradeArgs = @("--version")
}

Push-Location -LiteralPath $ProjectRoot
try {
    & $Python -m freqtrade @FreqtradeArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
