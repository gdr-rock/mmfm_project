param(
  [string]$Config = "configs/subset_example.yaml"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
  throw "Virtual environment not found. Run scripts/create_env.ps1 first."
}

& .\.venv\Scripts\python -m src.toc.main --config $Config
