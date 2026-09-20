param([switch]$DryRun, [string]$Text, [switch]$Execute, [double]$Duration = 0)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:NEEDLE_TELEMETRY = '0'
$env:PYTHONUNBUFFERED = '1'
$secretPath = Join-Path $PSScriptRoot 'data/openhab-password.dpapi'
$config = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'config.json') -Raw | ConvertFrom-Json
if (Test-Path -LiteralPath $secretPath) {
    $securePassword = Get-Content -LiteralPath $secretPath | ConvertTo-SecureString
    $credential = [PSCredential]::new($(if ($env:OPENHAB_USER) { $env:OPENHAB_USER } else { 'admin' }), $securePassword)
    $env:OPENHAB_USER = $credential.UserName
    $env:OPENHAB_PASSWORD = $credential.GetNetworkCredential().Password
}
$arguments = @('-m', 'jarvis')
if ($DryRun) { $arguments += '--dry-run' }
if ($Text) { $arguments += @('--text', $Text) }
if ($Execute) { $arguments += '--execute' }
if ($Duration -gt 0) { $arguments += @('--duration', "$Duration") }
try { & "$PSScriptRoot\.venv\Scripts\python.exe" @arguments }
finally { Remove-Item Env:OPENHAB_PASSWORD -ErrorAction SilentlyContinue }
