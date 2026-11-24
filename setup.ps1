# One-time setup on Windows (PowerShell). Needs Python 3.10 on PATH as `py -3.10`; Node is optional.
# Creates .venv-gpu (research + inference) and .venv-app (dashboard), then fetches public references and HD-BET weights.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path .venv-gpu)) { py -3.10 -m venv .venv-gpu }
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path .venv-app)) { py -3.10 -m venv .venv-app }
.\.venv-app\Scripts\python.exe -m pip install --upgrade pip
.\.venv-app\Scripts\python.exe -m pip install -r app\requirements.txt

# MNI152NLin2009cAsym template + Harvard-Oxford ROI atlas (public, ~30 MB)
.\.venv-gpu\Scripts\python.exe src\fetch_references.py
# HD-BET skull-stripping weights (public, ~120 MB)
.\.venv-gpu\Scripts\python.exe download_hd_bet_weights.py

Write-Host "`nSetup complete. Start the app with:  .\run_dashboard.bat" -ForegroundColor Green
