@echo off
REM Create GitHub repo and push — run AFTER: gh auth login
set REPO=digital-banking-security-trust
set USER=gracious0102
cd /d "%~dp0"

echo Checking GitHub auth...
gh auth status
if errorlevel 1 (
    echo.
    echo Not logged in. Run: gh auth login
    echo Then run this script again.
    exit /b 1
)

echo Creating repo %USER%/%REPO% on GitHub...
gh repo create %USER%/%REPO% --public --source=. --remote=origin --description "Digital banking security, privacy and trust — 199-respondent survey analysis with policy recommendations" --push

if errorlevel 1 (
    echo Repo may already exist. Adding remote and pushing...
    git remote add origin https://github.com/%USER%/%REPO%.git 2>nul
    git branch -M main
    git push -u origin main
)

echo Done. View at: https://github.com/%USER%/%REPO%
pause
