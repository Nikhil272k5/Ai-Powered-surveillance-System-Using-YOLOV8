#!/usr/bin/env python3
"""
AbnoGuard - AI Surveillance for Abnormality & Abandoned Objects
Main entry point using YOLOv8 for enhanced detection
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
from video_runner import VideoRunner
from utils import setup_directories, check_dependencies

def main():
    """Main entry point for AbnoGuard"""
    print("🚀 AbnoGuard - AI Surveillance System")
    print("🔍 Powered by YOLOv8 for Enhanced Detection")
    print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependencies check failed. Please install required packages.")
        return
    
    # Create Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Show file picker dialog
    print("📁 Please select a video file for analysis...")
    video_path = filedialog.askopenfilename(
        title="Select Video File for AbnoGuard Analysis",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("All files", "*.*")
        ]
    )
    
    if not video_path:
        print("❌ No video file selected. Exiting.")
        return
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"✅ Selected video: {video_path}")
    print(f"📊 File size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
    
    # Run video analysis
    try:
        print("\n🎬 Starting video analysis with YOLOv8...")
        print("💡 YOLOv8 provides better accuracy and more object classes")
        
        runner = VideoRunner(video_path)
        runner.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Error during video analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 AbnoGuard analysis complete!")
    print("📁 Check the 'outputs/' folder for results and alerts")

if __name__ == "__main__":
    main()

