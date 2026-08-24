$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$ok = $true

function Check($name, $cmd) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) { Write-Host "[OK] $name" }
    else { Write-Host "[MISSING] $name"; $script:ok = $false }
}

Check "Python" "python"
Check "Tesseract" "tesseract"

if (Test-Path "$root\.env.local") { Write-Host "[OK] .env.local" }
else { Write-Host "[INFO] Copy .env.local.example to .env.local" }

if (-not $ok) { exit 1 }
