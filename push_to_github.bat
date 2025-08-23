@echo off
echo 🚀 AbnoGuard GitHub Repository Update Script
echo ================================================
echo.

echo 📁 Current directory: %CD%
echo.

echo 🔍 Checking if this is a Git repository...
if not exist ".git" (
    echo ❌ This is not a Git repository!
    echo.
    echo 📋 Please run these commands manually:
    echo.
    echo 1. Navigate to your GitHub repository folder:
    echo    cd C:\path\to\your\github\repo
    echo.
    echo 2. Run the script from there
    echo.
    pause
    exit /b 1
)

echo ✅ Git repository found!
echo.

echo 📥 Adding all files to Git...
git add .

echo.
echo 📝 Committing changes...
git commit -m "🚀 Complete AbnoGuard system upgrade with YOLOv8

- Upgraded from YOLOv5 to YOLOv8 for better detection
- Added comprehensive object tracking system
- Implemented abandoned object detection
- Added abnormal movement detection (speed spike, loitering, counterflow)
- Created professional README with documentation
- Added debug tools and testing scripts
- Optimized for real-time surveillance applications"

echo.
echo 🚀 Pushing to GitHub...
git push origin main

echo.
echo ✅ All files have been pushed to GitHub!
echo.
echo 🌟 Your repository is now updated with the complete AbnoGuard system!
echo.
pause
