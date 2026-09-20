param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$pidPath = Join-Path $PSScriptRoot 'data/jarvis.pid'
if (Test-Path -LiteralPath $pidPath) {
    $existingId = [int](Get-Content -LiteralPath $pidPath)
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingId" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*$PSScriptRoot*run.ps1*") {
        Write-Output "Jarvis is already running (PID $existingId)."
        exit
    }
}
New-Item -ItemType Directory -Force -Path "$PSScriptRoot\logs", "$PSScriptRoot\data" | Out-Null
$launchArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\run.ps1`""
if ($DryRun) { $launchArgs += ' -DryRun' }
$shellPath = (Get-Process -Id $PID).Path
$worker = Start-Process $shellPath -ArgumentList $launchArgs -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "$PSScriptRoot\logs\stdout.log" -RedirectStandardError "$PSScriptRoot\logs\stderr.log"
$worker.Id | Set-Content -LiteralPath $pidPath
Write-Output "Jarvis started (PID $($worker.Id)). Logs: $PSScriptRoot\logs\jarvis.log"
