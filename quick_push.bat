@echo off
echo 🚀 Quick Push to GitHub - Starting NOW!
echo.

cd /d C:\Users\Nikhil
if exist "Ai-Powered-surveillance-System-Using-YOLOV8" (
    echo 📁 Repository folder exists - removing old one...
    rmdir /s /q "Ai-Powered-surveillance-System-Using-YOLOV8"
)

echo 📥 Cloning repository...
git clone https://github.com/Nikhil272k5/Ai-Powered-surveillance-System-Using-YOLOV8.git
cd Ai-Powered-surveillance-System-Using-YOLOV8

echo 📋 Copying all AbnoGuard files...
xcopy "C:\Users\Nikhil\abnoguard\*" "." /E /H /Y /Q

echo 🚀 Pushing to GitHub...
git add .
git commit -m "🚀 Complete AbnoGuard System with YOLOv8"
git push origin main

echo ✅ DONE! All files pushed to GitHub!
echo 🌟 Check: https://github.com/Nikhil272k5/Ai-Powered-surveillance-System-Using-YOLOV8
pause
