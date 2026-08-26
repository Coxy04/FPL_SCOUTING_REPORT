# Refreshes the FPL Scouting Report: refetches data, retrains, rebuilds the dashboard,
# and pushes it to GitHub Pages. Run from PowerShell: .\refresh_dashboard.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Running model (fetch + train + predict)..." -ForegroundColor Cyan
& ".\fpl_env\Scripts\python.exe" fpl_ml_model.py
if ($LASTEXITCODE -ne 0) { throw "fpl_ml_model.py failed" }

Write-Host "`nBuilding dashboard..." -ForegroundColor Cyan
& ".\fpl_env\Scripts\python.exe" build_dashboard.py
if ($LASTEXITCODE -ne 0) { throw "build_dashboard.py failed" }

Write-Host "`nPushing to GitHub Pages..." -ForegroundColor Cyan
git add docs/index.html
git commit -m "Refresh dashboard data"
git push

Write-Host "`nDone. Live at https://coxy04.github.io/FPL_SCOUTING_REPORT/" -ForegroundColor Green
