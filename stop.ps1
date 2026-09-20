$ErrorActionPreference = 'Stop'
$pidPath = Join-Path $PSScriptRoot 'data/jarvis.pid'
if (!(Test-Path -LiteralPath $pidPath)) { Write-Output 'Jarvis is not running.'; exit }
$workerId = [int](Get-Content -LiteralPath $pidPath)
$worker = Get-CimInstance Win32_Process -Filter "ProcessId = $workerId" -ErrorAction SilentlyContinue
if ($worker -and $worker.CommandLine -like "*$PSScriptRoot*run.ps1*") {
    # Kill only this verified launcher and its child Python process tree.
    & taskkill.exe /PID $workerId /T /F
}
Remove-Item -LiteralPath $pidPath
