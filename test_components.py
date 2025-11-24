#!/usr/bin/env python3
"""
Test script to verify that all components of the Grievance System work correctly
"""

def test_database():
    """Test database functionality"""
    try:
        from database import DatabaseManager
        db = DatabaseManager()
        print("✅ Database initialized successfully")
        
        # Test user creation
        result = db.create_user("testuser", "test@example.com", "password123", "student")
        print(f"✅ User creation test: {'Success' if result else 'User already exists'}")
        
        # Test authentication
        user = db.authenticate_user("admin", "admin123")
        if user:
            print(f"✅ Authentication test successful - Admin user: {user['username']}")
        else:
            print("❌ Authentication test failed")
            
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_sentiment_analyzer():
    """Test sentiment analyzer functionality"""
    try:
        from sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        print("✅ Sentiment Analyzer initialized successfully")
        
        # Test sentiment analysis
        test_texts = [
            "I am very happy with the service!",
            "This is an urgent issue that needs immediate attention",
            "The food quality is okay, nothing special"
        ]
        
        for text in test_texts:
            analysis = analyzer.analyze_sentiment(text)
            print(f"✅ Text: '{text[:30]}...'")
            print(f"   Sentiment: {analysis['sentiment']} {analyzer.get_sentiment_emoji(analysis['sentiment'])}")
            print(f"   Priority: {analyzer.get_priority_label(analysis['priority'])}")
            print(f"   Confidence: {analysis['confidence']:.0%}")
            print()
            
        return True
    except Exception as e:
        print(f"❌ Sentiment Analyzer test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔬 Testing Grievance System Components\n")
    print("=" * 50)
    
    db_success = test_database()
    print("\n" + "=" * 50)
    sa_success = test_sentiment_analyzer()
    
    print("\n" + "=" * 50)
    if db_success and sa_success:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo run the application, use:")
        print("streamlit run main_app.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
