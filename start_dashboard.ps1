param([switch]$Setup, [switch]$Check, [int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'run_dashboard.py'
$arguments = @($launcher, '--port', "$Port")
if ($Setup) { $arguments += '--setup' }
if ($Check) { $arguments += '--check' }
if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 @arguments }
elseif (Get-Command python -ErrorAction SilentlyContinue) { & python @arguments }
else { throw 'Install Python 3.11 or 3.12, then run this command again.' }
exit $LASTEXITCODE
