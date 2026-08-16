param(
    [string]$Container = "freqtrade-hedge-HPRL",
    [string]$ProjectRoot = "/opt/freqtrade-hedge",
    [string]$Python = "/opt/hedge-venv/bin/python",
    [string]$DsnEnvironmentVariable = "HPRL_POSTGRES_DSN",
    [string]$Output = "/opt/freqtrade-hedge/user_data/production-readiness/hprl-r2-postgres-probes.json"
)

$ErrorActionPreference = "Stop"
Set-ExecutionPolicy -Scope Process Bypass -Force
Write-Host "=== HPRL V3 Production R2 PostgreSQL probes ==="
Write-Host "This runner performs bounded probe writes in the dedicated probe schema."
Write-Host "It does not claim failover or backup/restore completion."

$ProbeCode = "import os,sys; sys.exit(0 if os.environ.get('$DsnEnvironmentVariable','').strip() else 2)"
& docker exec $Container $Python -c $ProbeCode
if ($LASTEXITCODE -ne 0) { throw "Container environment variable is missing: $DsnEnvironmentVariable" }

& docker exec -e PYTHONDONTWRITEBYTECODE=1 -w $ProjectRoot $Container $Python tools/hprl_v3_production_r2.py postgres-probes --dsn-env $DsnEnvironmentVariable --output $Output
if ($LASTEXITCODE -ne 0) { throw "PostgreSQL basic/concurrency/durability probes failed." }
Write-Host "POSTGRES_RUNTIME_PROBES=PASS"
Write-Host "POSTGRES_FULL_R2=LOCKED (controlled failover and isolated backup/restore evidence still required)"
