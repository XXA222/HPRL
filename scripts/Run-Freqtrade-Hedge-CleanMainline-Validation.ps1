[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [switch]$FullOfflinePytest,
    [switch]$IncludeFullRuffInventory
)

$ErrorActionPreference = "Continue"
Set-StrictMode -Version Latest

function Step {
    param([string]$Name)
    Write-Host ""
    Write-Host ("=== " + $Name + " ===") -ForegroundColor Cyan
}

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

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$AuditRoot = Join-Path $ProjectRoot ("user_data\audit\clean-mainline-validation-" + $Timestamp)
$TempRoot = Join-Path $AuditRoot "temp"
$BaseTemp = Join-Path $AuditRoot "pytest-basetemp"
New-Item -ItemType Directory -Path $AuditRoot -Force | Out-Null
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null

$OldTemp = $env:TEMP
$OldTmp = $env:TMP
$OldTmpDir = $env:TMPDIR
$env:TEMP = $TempRoot
$env:TMP = $TempRoot
$env:TMPDIR = $TempRoot

$Results = [ordered]@{}

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Command,
        [bool]$Blocking = $true
    )
    Step $Name
    & $Command
    $Code = $LASTEXITCODE
    $Results[$Name] = [ordered]@{ ExitCode = $Code; Blocking = $Blocking }
    if ($Code -ne 0) {
        if ($Blocking) {
            Write-Warning ($Name + " failed with exit code " + $Code)
        }
        else {
            Write-Warning ($Name + " reported nonblocking findings; exit code " + $Code)
        }
    }
}

Push-Location -LiteralPath $ProjectRoot
try {
    Step "Local source authority"
    & $Python -c "import pathlib,sys,freqtrade,freqtrade_client; root=pathlib.Path.cwd().resolve(); client=(root/'ft_client').resolve(); fm=pathlib.Path(freqtrade.__file__).resolve(); cm=pathlib.Path(freqtrade_client.__file__).resolve(); print('Python:',sys.executable); print('Freqtrade:',fm); print('FreqtradeClient:',cm); print('Project:',root); assert root in fm.parents, (root,fm); assert client in cm.parents, (client,cm)"
    $Results["Local source authority"] = [ordered]@{ ExitCode = $LASTEXITCODE; Blocking = $true }

    Run-Step "pip check" { & $Python -m pip check }

    Run-Step "Clean mainline workspace validator" {
        & $Python tools\validate_clean_mainline.py `
            --project-root $ProjectRoot `
            --workspace-mode `
            --output (Join-Path $AuditRoot "clean-mainline-validation.json")
    }

    Run-Step "Clean Mainline 200-point matrix" {
        & $Python tools\validate_clean_mainline_200.py `
            --project-root $ProjectRoot `
            --workspace-mode `
            --output (Join-Path $AuditRoot "clean-mainline-200.json")
    }

    $ConfigBaseTemp = Join-Path $AuditRoot "pytest-config-basetemp"
    Run-Step "Clean config isolation pytest" {
        & $Python -m pytest -q -ra -o addopts= `
            --basetemp $ConfigBaseTemp `
            tests\hedge\test_clean_mainline_config_isolation.py `
            tests\hedge\operations\test_config.py `
            tests\hedge\persistence\test_central_integration.py
    }

    $UpstreamConfigBaseTemp = Join-Path $AuditRoot "pytest-upstream-config-basetemp"
    Run-Step "Upstream config isolation sentinels" {
        & $Python -m pytest -q -ra -o addopts= `
            --basetemp $UpstreamConfigBaseTemp `
            tests\test_configuration.py::test_validate_price_side `
            tests\plugins\test_pairlist.py::test_pairlistmanager_no_pairlist `
            tests\rpc\test_rpc.py::test_rpc_health `
            tests\test_wallets.py::test_sync_wallet_dry
    }

    Step "PowerShell 5.1 AST"
    $PowerShellFailures = @()
    Get-ChildItem -LiteralPath $ProjectRoot -Recurse -File -Filter "*.ps1" |
        Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\user_data\*" } |
        ForEach-Object {
            $Tokens = $null
            $Errors = $null
            [void][System.Management.Automation.Language.Parser]::ParseFile(
                $_.FullName,
                [ref]$Tokens,
                [ref]$Errors
            )
            if ($Errors.Count -gt 0) {
                $PowerShellFailures += $_.FullName
                $Errors | ForEach-Object { Write-Host $_ }
            }
        }
    $PsExit = if ($PowerShellFailures.Count -eq 0) { 0 } else { 1 }
    $global:LASTEXITCODE = $PsExit
    $Results["PowerShell 5.1 AST"] = [ordered]@{ ExitCode = $PsExit; Blocking = $true }

    & $Python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('ruff') else 3)"
    if ($LASTEXITCODE -eq 0) {
        Run-Step "Blocking Ruff correctness" {
            & $Python -m ruff check freqtrade\hedge freqtrade\freqai\hedge_rl tests\hedge tools scripts --select E9,F63,F7,F82
        }
        if ($IncludeFullRuffInventory) {
            Run-Step "Full Ruff debt inventory" {
                & $Python -m ruff check freqtrade\hedge freqtrade\freqai\hedge_rl tests\hedge tools scripts `
                    --output-format concise `
                    --output-file (Join-Path $AuditRoot "ruff-full-debt.txt")
            } $false
        }
    }
    else {
        Write-Warning "Ruff is not installed in the existing virtual environment; Ruff gates skipped."
        $Results["Blocking Ruff correctness"] = [ordered]@{ ExitCode = 3; Blocking = $false }
    }

    Run-Step "Research quality" {
        & $Python tools\validate_hedge_research_quality.py `
            --project-root $ProjectRoot `
            --output (Join-Path $AuditRoot "research-quality.json")
    }
    Run-Step "Research validation matrix" {
        & $Python tools\run_hedge_research_validation.py `
            --project-root $ProjectRoot `
            --output (Join-Path $AuditRoot "research-validation.json")
    }
    Run-Step "MLRL code quality" {
        & $Python tools\validate_hedge_mlrl_code_quality.py `
            --source $ProjectRoot `
            --json-out (Join-Path $AuditRoot "mlrl-quality.json")
    }
    Run-Step "MLRL validation matrix" {
        & $Python tools\run_hedge_mlrl_validation.py `
            --source $ProjectRoot `
            --json-out (Join-Path $AuditRoot "mlrl-validation.json")
    }
    Run-Step "Integrated Paper closed-Bar smoke" {
        & $Python tools\hedge_integrated_smoke.py
    }
    Run-Step "Full Hedge pytest" {
        & $Python -m pytest -q -ra -o addopts= --basetemp $BaseTemp tests\hedge
    }

    if ($FullOfflinePytest) {
        $FullBaseTemp = Join-Path $AuditRoot "pytest-full-basetemp"
        Run-Step "Full offline pytest" {
            & $Python -m pytest -q -ra -o addopts= `
                --basetemp $FullBaseTemp `
                --ignore=tests\test_pip_audit.py `
                --ignore=tests\exchange_online `
                tests
        }
    }
}
finally {
    $env:TEMP = $OldTemp
    $env:TMP = $OldTmp
    if ($null -eq $OldTmpDir) {
        Remove-Item Env:TMPDIR -ErrorAction SilentlyContinue
    }
    else {
        $env:TMPDIR = $OldTmpDir
    }
    Pop-Location
}

$BlockingFailures = @(
    $Results.GetEnumerator() |
        Where-Object { $_.Value.Blocking -and $_.Value.ExitCode -ne 0 }
)

$Summary = [ordered]@{
    Schema = "freqtrade-hedge-clean-mainline-windows-validation-v1"
    ProjectRoot = $ProjectRoot
    Python = $Python
    AuditRoot = $AuditRoot
    BlockingFailures = $BlockingFailures.Count
    Status = if ($BlockingFailures.Count -eq 0) { "PASS" } else { "FAIL" }
    Results = $Results
}
$SummaryPath = Join-Path $AuditRoot "summary.json"
$Summary | ConvertTo-Json -Depth 8 | Out-File -LiteralPath $SummaryPath -Encoding utf8

Step "FINAL SUMMARY"
$Summary | ConvertTo-Json -Depth 8
if ($BlockingFailures.Count -eq 0) {
    Write-Host "CLEAN MAINLINE WINDOWS GATE: PASS" -ForegroundColor Green
    exit 0
}
Write-Host "CLEAN MAINLINE WINDOWS GATE: FAIL" -ForegroundColor Red
exit 1
