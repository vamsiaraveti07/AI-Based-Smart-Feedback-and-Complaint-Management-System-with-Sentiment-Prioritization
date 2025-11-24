"""
Startup script for the Enhanced AI-Powered Grievance System
This script handles initialization and runs the main application
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Error installing dependencies: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Installation timed out. Please install manually with: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"⚠️ Warning: Could not download NLTK data: {e}")
        return True  # Continue anyway

def initialize_database():
    """Initialize the database"""
    print("\n🗄️ Initializing database...")
    try:
        from database import DatabaseManager
        db_manager = DatabaseManager()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        'enhanced_grievance_app.py',
        'database.py',
        'advanced_sentiment_analyzer.py',
        'smart_routing_system.py',
        'root_cause_analyzer.py',
        'emotion_aware_chatbot.py',
        'resolution_quality_predictor.py',
        'anonymous_trust_system.py',
        'mood_tracker.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def run_streamlit_app():
    """Run the Streamlit application"""
    print("\n🚀 Starting the Enhanced Grievance System...")
    print("📝 The application will open in your default web browser")
    print("🌐 Default URL: http://localhost:8501")
    print("\n" + "="*50)
    print("🎯 ENHANCED AI-POWERED GRIEVANCE SYSTEM")
    print("="*50)
    print("Features included:")
    print("• 🧠 Emotion-Aware AI Analysis")
    print("• 🎯 Smart Auto-Routing")
    print("• 🔍 Root Cause Detection")
    print("• 🤖 Intelligent Chatbot")
    print("• 📊 Quality Prediction")
    print("• 🕵️ Anonymous Trust System")
    print("• 📈 Mood Tracking")
    print("• 📊 Advanced Analytics")
    print("="*50)
    print("\n⚠️  To stop the application, press Ctrl+C in this terminal")
    print("\n")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "enhanced_grievance_app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        print("\n🔧 Try running manually with: streamlit run enhanced_grievance_app.py")

def main():
    """Main startup function"""
    print("🎯 Enhanced AI-Powered Grievance System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check if files exist
    if not check_files():
        print("\n❌ Setup incomplete. Please ensure all files are present.")
        return
    
    # Install dependencies
    install_choice = input("\n❓ Install/update dependencies? (y/n, default=y): ").lower()
    if install_choice != 'n':
        if not install_dependencies():
            print("\n❌ Failed to install dependencies. Please run manually:")
            print("   pip install -r requirements.txt")
            return
    
    # Download NLTK data
    download_nltk_data()
    
    # Initialize database
    if not initialize_database():
        print("\n❌ Database initialization failed")
        return
    
    print("\n✅ Setup complete! Starting application...")
    time.sleep(2)
    
    # Run the application
    run_streamlit_app()

if __name__ == "__main__":
    main()
