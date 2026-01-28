"""
ABNOGUARD - AI Surveillance System
Main entry point - runs the dashboard server
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Start the AbnoGuard surveillance system"""
    print("=" * 60)
    print("🛡️  ABNOGUARD - AI SURVEILLANCE SYSTEM")
    print("=" * 60)
    print()
    print("📌 Upload a video to start analysis")
    print("🌐 Dashboard: http://127.0.0.1:8000")
    print()
    
    from dashboard.backend.main import start_server
    start_server()

if __name__ == "__main__":
    main()
