[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ConfigPath = "user_data\config_hedge.json",
    [string]$Strategy = "",
    [string]$Image = "freqtrade-hedge:1.3.1-runtime",
    [string]$ContainerName = "freqtrade-hedge-runtime"
)

$ErrorActionPreference = "Stop"

$DockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if (-not $DockerCommand) {
    throw "Docker CLI was not found. Install/start Docker Desktop first."
}
& docker version --format "{{.Server.Version}}" *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop exists, but the Docker Engine is not running."
}

$Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$UserData = Join-Path $Root "user_data"
if (-not (Test-Path -LiteralPath $UserData -PathType Container)) {
    New-Item -ItemType Directory -Path $UserData -Force | Out-Null
}
$UserData = (Resolve-Path -LiteralPath $UserData).Path
$Config = (Resolve-Path -LiteralPath (Join-Path $Root $ConfigPath)).Path

$UserDataPrefix = $UserData.TrimEnd('\') + '\'
if (-not $Config.StartsWith($UserDataPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "The runtime config must be located under $UserData"
}

$Data = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
if ($Data.dry_run -ne $true) {
    throw "Docker runtime is locked to dry_run=true in this release."
}
if (-not $Data.hedge -or $Data.hedge.read_only -ne $true) {
    throw "Docker runtime requires hedge.read_only=true."
}
if ($Data.hedge.live_trading_enabled -ne $false) {
    throw "Docker runtime requires hedge.live_trading_enabled=false."
}
if ($Data.hedge.operation_mode -notin @("paper", "readonly", "read_only")) {
    throw "Docker runtime requires paper/readonly operation_mode."
}

& docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Validated runtime image does not exist: $Image"
}

$Existing = & docker ps -a --filter "name=^/$ContainerName$" --format "{{.ID}}"
if ($Existing) {
    & docker rm -f $ContainerName *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to remove the previous container: $ContainerName"
    }
}

$RelativeConfig = $Config.Substring($UserDataPrefix.Length).Replace('\', '/')
$ContainerConfig = "/freqtrade/user_data/$RelativeConfig"
$Args = @(
    "run", "--detach", "--init",
    "--name", $ContainerName,
    "--security-opt", "no-new-privileges:true",
    "--env", "FT_APP_ENV=docker",
    "--env", "PYTHONUNBUFFERED=1",
    "--publish", "127.0.0.1:8080:8080",
    "--volume", "${UserData}:/freqtrade/user_data",
    $Image,
    "trade", "--config", $ContainerConfig
)
if ($Strategy) {
    $Args += @("--strategy", $Strategy)
}

& docker @Args
if ($LASTEXITCODE -ne 0) {
    throw "Failed to start Docker runtime."
}

Write-Host "Hedge runtime started inside Docker: $ContainerName" -ForegroundColor Green
Write-Host "Image: $Image" -ForegroundColor Cyan
Write-Host "Config: $ContainerConfig" -ForegroundColor Cyan
Write-Host "Logs: docker logs -f $ContainerName" -ForegroundColor Cyan
