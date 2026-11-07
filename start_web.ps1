$ErrorActionPreference = 'Stop'
$serverPython = Join-Path $PSScriptRoot '.venv_web/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $serverPython)) {
    $serverPython = (Get-Command python -ErrorAction Stop).Source
}
& $serverPython (Join-Path $PSScriptRoot 'run_web.py') @args
exit $LASTEXITCODE
