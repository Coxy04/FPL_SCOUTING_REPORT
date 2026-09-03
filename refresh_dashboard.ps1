# Refreshes the FPL Scouting Report end to end: self-tunes, then refetches data, retrains, rebuilds
# the dashboard, and pushes it to GitHub Pages. Manual only -- run whenever you want fresh data,
# there's no scheduled autorun for this. From PowerShell: .\refresh_dashboard.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Backtesting current accuracy..." -ForegroundColor Cyan
& ".\fpl_env\Scripts\python.exe" backtest_model.py
if ($LASTEXITCODE -ne 0) { throw "backtest_model.py failed" }

Write-Host "`nSearching for better hyperparameters (--apply)..." -ForegroundColor Cyan
& ".\fpl_env\Scripts\python.exe" tune_model.py --apply
if ($LASTEXITCODE -ne 0) { throw "tune_model.py failed" }

Write-Host "`nRunning model (fetch + train + predict)..." -ForegroundColor Cyan
& ".\fpl_env\Scripts\python.exe" fpl_ml_model.py
if ($LASTEXITCODE -ne 0) { throw "fpl_ml_model.py failed" }

Write-Host "`nScoring past picks / picking this gameweek's team..." -ForegroundColor Cyan
& ".\fpl_env\Scripts\python.exe" pick_team.py
if ($LASTEXITCODE -ne 0) { throw "pick_team.py failed" }

Write-Host "`nRating my real team and checking transfers..." -ForegroundColor Cyan
& ".\fpl_env\Scripts\python.exe" fetch_my_team.py
if ($LASTEXITCODE -ne 0) { throw "fetch_my_team.py failed" }

Write-Host "`nBuilding dashboard..." -ForegroundColor Cyan
& ".\fpl_env\Scripts\python.exe" build_dashboard.py
if ($LASTEXITCODE -ne 0) { throw "build_dashboard.py failed" }

Write-Host "`nPushing to GitHub Pages..." -ForegroundColor Cyan
git add fpl_ml_model.py fpl_ml_accuracy_history.jsonl fpl_ml_team_history.jsonl prediction_history docs/index.html

$staged = git diff --cached --name-only
if ($staged) {
    git commit -m "Refresh dashboard data: $(Get-Date -Format 'yyyy-MM-dd')"
    git push
    Write-Host "`nDone. Live at https://coxy04.github.io/FPL_SCOUTING_REPORT/" -ForegroundColor Green
} else {
    Write-Host "`nNo changes to push (data identical to last run)." -ForegroundColor Yellow
}
