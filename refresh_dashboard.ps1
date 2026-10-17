# Refreshes the FPL Scouting Report end to end: backtests, retrains, re-picks, rebuilds the dashboard
# and pushes it to GitHub Pages. Manual only -- run whenever you want fresh data (ideally the morning
# of every deadline); there's no scheduled autorun for this. From PowerShell: .\refresh_dashboard.ps1
#
# Hyperparameter tuning runs at most once every TUNE_EVERY_DAYS. It used to run on every refresh, which
# meant a random search over a static archive could re-pick different settings on identical data and
# reshuffle the whole ranking for no reason (see SEASON_REVIEW_GW1-6.md, "why the algorithm changes").
# A weekly refresh now only retrains on new data; the settings stay put between monthly tunes.
# Force a tune with: .\refresh_dashboard.ps1 -Tune
param([switch]$Tune)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$TUNE_EVERY_DAYS = 28

Write-Host "Backtesting current accuracy..." -ForegroundColor Cyan
& ".\fpl_env\Scripts\python.exe" backtest_model.py
if ($LASTEXITCODE -ne 0) { throw "backtest_model.py failed" }

$tuneDue = $Tune
if (-not $tuneDue) {
    if (Test-Path "fpl_ml_tuning.json") {
        $tuning = Get-Content "fpl_ml_tuning.json" -Raw | ConvertFrom-Json
        $last = $null
        if ($tuning.PSObject.Properties.Name -contains "generated_at") { $last = [datetime]$tuning.generated_at }
        if ($null -eq $last -or ((Get-Date).ToUniversalTime() - $last.ToUniversalTime()).TotalDays -ge $TUNE_EVERY_DAYS) { $tuneDue = $true }
    } else { $tuneDue = $true }
}
if ($tuneDue) {
    Write-Host "`nSearching for better hyperparameters (--apply, due every $TUNE_EVERY_DAYS days)..." -ForegroundColor Cyan
    & ".\fpl_env\Scripts\python.exe" tune_model.py --apply
    if ($LASTEXITCODE -ne 0) { throw "tune_model.py failed" }
} else {
    Write-Host "`nSkipping hyperparameter tuning (last tuned within $TUNE_EVERY_DAYS days; pass -Tune to force)." -ForegroundColor DarkGray
}

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
git add fpl_ml_model.py fpl_ml_accuracy_history.jsonl fpl_ml_team_history.jsonl my_fpl_team_history.jsonl prediction_history docs/index.html

$staged = git diff --cached --name-only
if ($staged) {
    git commit -m "Refresh dashboard data: $(Get-Date -Format 'yyyy-MM-dd')"
    # git writes its push summary ("To https://...") to stderr even on success, which Windows
    # PowerShell 5.1 turns into a NativeCommandError under $ErrorActionPreference = "Stop" -- the
    # push had succeeded but the script reported failure. --porcelain puts the summary on stdout;
    # the exit code is still the real signal.
    git push --porcelain
    if ($LASTEXITCODE -ne 0) { throw "git push failed" }
    Write-Host "`nDone. Live at https://coxy04.github.io/FPL_SCOUTING_REPORT/" -ForegroundColor Green
} else {
    Write-Host "`nNo changes to push (data identical to last run)." -ForegroundColor Yellow
}
