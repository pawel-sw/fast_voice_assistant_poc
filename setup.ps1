$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (!(Test-Path -LiteralPath config.json)) { Copy-Item -LiteralPath config.json.sample -Destination config.json }
if (!(Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.sample -Destination .env }
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Installation failed with exit code $LASTEXITCODE" } }
if (!(Test-Path .venv\Scripts\python.exe)) { python -m venv .venv; Check-Exit }
& .venv\Scripts\python.exe -m pip install -r requirements-client.txt
Check-Exit
& .venv\Scripts\python.exe -m jarvis.wake
Check-Exit
