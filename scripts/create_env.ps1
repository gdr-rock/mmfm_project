param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

if (Test-Path .\.venv) {
  Write-Host ".venv already exists"
} else {
  & $Python -m venv .venv
  Write-Host "Created .venv"
}

& .\.venv\Scripts\python -m pip install --upgrade pip
Write-Host "Environment ready"
