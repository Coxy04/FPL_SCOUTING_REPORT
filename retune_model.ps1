# Weekly self-tuning run: backtests current accuracy, appends it to the running history, then
# searches for better per-position hyperparameters and applies any genuine improvement (>1% MAE,
# averaged across two holdout folds -- see tune_model.py). Every change is a normal git commit,
# so a bad update is always a `git revert` away. Meant to run unattended via Task Scheduler.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Running backtest (accuracy check)..." -ForegroundColor Cyan
& ".\fpl_env\Scripts\python.exe" backtest_model.py
if ($LASTEXITCODE -ne 0) { throw "backtest_model.py failed" }

Write-Host "`nRunning hyperparameter search (--apply)..." -ForegroundColor Cyan
& ".\fpl_env\Scripts\python.exe" tune_model.py --apply
if ($LASTEXITCODE -ne 0) { throw "tune_model.py failed" }

git add fpl_ml_model.py fpl_ml_accuracy_history.jsonl

$staged = git diff --cached --name-only
if ($staged) {
    Write-Host "`nChanges found, committing..." -ForegroundColor Cyan
    git commit -m "Weekly self-tune: $(Get-Date -Format 'yyyy-MM-dd')"
    git push
    Write-Host "Pushed." -ForegroundColor Green
} else {
    Write-Host "`nNo changes this week (no hyperparameters improved past the threshold)." -ForegroundColor Yellow
}
