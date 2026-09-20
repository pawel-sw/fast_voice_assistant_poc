param([switch]$Timings, [switch]$Detailed)
Set-Location -LiteralPath $PSScriptRoot
$arguments = @('-m', 'jarvis.logview')
if ($Timings) { $arguments += '--timings' }
if ($Detailed) { $arguments += '--detailed' }
& "$PSScriptRoot\.venv\Scripts\python.exe" @arguments
