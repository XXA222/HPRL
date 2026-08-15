[CmdletBinding()]
param([string]$ContainerName = "freqtrade-hedge-runtime")

$ErrorActionPreference = "Stop"
$Existing = & docker ps -a --filter "name=^/$ContainerName$" --format "{{.ID}}"
if (-not $Existing) {
    Write-Host "Container does not exist: $ContainerName" -ForegroundColor Yellow
    exit 0
}
& docker rm -f $ContainerName *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to stop Docker runtime: $ContainerName"
}
Write-Host "Docker runtime stopped: $ContainerName" -ForegroundColor Green
