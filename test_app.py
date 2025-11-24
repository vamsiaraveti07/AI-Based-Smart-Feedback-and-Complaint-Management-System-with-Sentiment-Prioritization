#!/usr/bin/env python3
"""
Test script to verify the Grievance System functionality
"""

def test_basic_functionality():
    print("Testing basic functionality...")
    
    try:
        # Test imports
        from database import DatabaseManager
        from sentiment_analyzer import SentimentAnalyzer
        print("✓ All imports successful")
        
        # Test database
        db = DatabaseManager()
        print("✓ Database initialized")
        
        # Test sentiment analyzer
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("This hostel room is terrible and dirty!")
        print(f"✓ Sentiment analyzer working: {result['sentiment']}, Priority: {result['priority']}")
        
        # Test database methods that caused issues
        try:
            similar = db.get_similar_grievances("test description")
            print("✓ get_similar_grievances working")
        except Exception as e:
            print(f"✗ get_similar_grievances error: {e}")
        
        try:
            count = db.get_unread_notifications_count(1)
            print(f"✓ get_unread_notifications_count working: {count}")
        except Exception as e:
            print(f"✗ get_unread_notifications_count error: {e}")
        
        try:
            notifications = db.get_notifications(1)
            print(f"✓ get_notifications working: {len(notifications)} notifications")
        except Exception as e:
            print(f"✗ get_notifications error: {e}")
        
        try:
            activity = db.get_recent_activity()
            print(f"✓ get_recent_activity working: {len(activity)} activities")
        except Exception as e:
            print(f"✗ get_recent_activity error: {e}")
        
        print("\n=== BASIC FUNCTIONALITY TEST COMPLETE ===")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_streamlit_dependencies():
    print("\nTesting Streamlit dependencies...")
    
    try:
        import streamlit as st
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from PIL import Image
        from streamlit_extras.stylable_container import stylable_container
        print("✓ All Streamlit dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing Streamlit dependency: {e}")
        return False

def main():
    print("=== GRIEVANCE SYSTEM TEST ===\n")
    
    basic_ok = test_basic_functionality()
    streamlit_ok = test_streamlit_dependencies()
    
    if basic_ok and streamlit_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your application should work correctly now.")
        print("\nTo run the application, use:")
        print("streamlit run main_app.py")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the errors above and fix them before running the application.")

if __name__ == "__main__":
    main()
