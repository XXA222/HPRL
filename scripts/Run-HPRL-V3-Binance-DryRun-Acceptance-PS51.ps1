param(
    [string]$Container = "freqtrade-hedge-HPRL",
    [string]$ProjectRoot = "/opt/freqtrade-hedge",
    [string]$Python = "/opt/hedge-venv/bin/python",
    [string]$Telemetry = "/opt/freqtrade-hedge/user_data/dryrun/telemetry.jsonl",
    [string]$Output = "/opt/freqtrade-hedge/user_data/production-readiness/hprl-r2-binance-dryrun.json",
    [int]$MinimumCycles = 100,
    [int]$MinimumMinutes = 30,
    [int]$MaximumGapSeconds = 300
)

$ErrorActionPreference = "Stop"
Set-ExecutionPolicy -Scope Process Bypass -Force
Write-Host "=== HPRL V3 Binance dry-run acceptance ==="
Write-Host "Safety model: real Binance market data + simulated execution + zero exchange write capability."

& docker exec $Container test -f $Telemetry
if ($LASTEXITCODE -ne 0) { throw "Dry-run telemetry not found: $Telemetry" }
& docker exec -e PYTHONDONTWRITEBYTECODE=1 -w $ProjectRoot $Container $Python tools/hprl_v3_production_r2.py binance-dryrun --telemetry $Telemetry --output $Output --minimum-cycles $MinimumCycles --minimum-minutes $MinimumMinutes --maximum-gap-seconds $MaximumGapSeconds
if ($LASTEXITCODE -ne 0) { throw "Binance dry-run acceptance failed." }
Write-Host "BINANCE_DRYRUN_ACCEPTANCE=PASS"
Write-Host "LIVE_EXCHANGE_WRITE=DISABLED"
