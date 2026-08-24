param(
    [switch]$SkipRedis
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent

Write-Host "Checking dependencies..."
& "$root\scripts\check-deps.ps1"

if (-not (Test-Path "$root\.venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv "$root\.venv"
}

& "$root\.venv\Scripts\Activate.ps1"
pip install -q -r "$root\requirements.txt"

Set-Location $root
Write-Host "Starting PII Redaction Portal (NiceGUI + API)"
Write-Host "  UI:  http://127.0.0.1:8000/"
Write-Host "  API: http://127.0.0.1:8000/docs"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
