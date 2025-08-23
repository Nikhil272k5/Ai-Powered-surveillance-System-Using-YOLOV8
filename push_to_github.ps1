# 🚀 AbnoGuard GitHub Repository Update Script
# PowerShell Version

Write-Host "🚀 AbnoGuard GitHub Repository Update Script" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📁 Current directory: $PWD" -ForegroundColor Yellow
Write-Host ""

# Check if this is a Git repository
Write-Host "🔍 Checking if this is a Git repository..." -ForegroundColor Yellow
if (-not (Test-Path ".git")) {
    Write-Host "❌ This is not a Git repository!" -ForegroundColor Red
    Write-Host ""
    Write-Host "📋 Please run these commands manually:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Navigate to your GitHub repository folder:" -ForegroundColor White
    Write-Host "   cd C:\path\to\your\github\repo" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Run the script from there" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host "✅ Git repository found!" -ForegroundColor Green
Write-Host ""

# Check Git status
Write-Host "📊 Checking Git status..." -ForegroundColor Yellow
git status

Write-Host ""
Write-Host "📥 Adding all files to Git..." -ForegroundColor Yellow
git add .

Write-Host ""
Write-Host "📝 Committing changes..." -ForegroundColor Yellow
$commitMessage = @"
🚀 Complete AbnoGuard system upgrade with YOLOv8

- Upgraded from YOLOv5 to YOLOv8 for better detection
- Added comprehensive object tracking system
- Implemented abandoned object detection
- Added abnormal movement detection (speed spike, loitering, counterflow)
- Created professional README with documentation
- Added debug tools and testing scripts
- Optimized for real-time surveillance applications
"@

git commit -m $commitMessage

Write-Host ""
Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Yellow
git push origin main

Write-Host ""
Write-Host "✅ All files have been pushed to GitHub!" -ForegroundColor Green
Write-Host ""
Write-Host "🌟 Your repository is now updated with the complete AbnoGuard system!" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to continue"
