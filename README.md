# 🎯 Advanced AI-Powered Grievance System

A comprehensive, intelligent grievance management system with advanced AI capabilities for emotion analysis, smart routing, root cause detection, and organizational mood tracking.

## 🌟 Features Overview

This enhanced grievance system includes **12 advanced AI-powered features** that revolutionize complaint management:

### 1. 🧠 Emotion-Aware Sentiment Analyzer
- **Advanced NLP Models**: Uses BERT/RoBERTa for sophisticated emotion detection
- **Multi-Emotion Detection**: Identifies anger, sadness, anxiety, sarcasm, and frustration
- **Intensity Analysis**: Measures emotional intensity and confidence levels
- **Sarcasm Detection**: Specialized detection of hidden frustration through sarcasm
- **Impact**: Ensures emotionally urgent grievances are prioritized appropriately

### 2. 📊 Impact Score Generator
- **Weighted Scoring Algorithm**: Considers emotion, complaint length, keyword severity, and past history
- **Dynamic Calculation**: Real-time assessment of grievance impact
- **Historical Context**: Incorporates user's complaint history for better scoring
- **Priority Assignment**: Automatically assigns priority levels based on comprehensive analysis
- **Impact**: Highlights systemically significant issues over general feedback

### 3. 🎯 Smart Auto-Routing System
- **Multi-Factor Analysis**: Routes based on category, keywords, staff performance, and workload
- **Load Balancing**: Distributes cases evenly across available staff
- **Specialization Matching**: Assigns cases to staff with relevant expertise
- **Performance Tracking**: Monitors routing effectiveness and adjusts accordingly
- **Impact**: Reduces response time and prevents misrouted tickets

### 4. 🔍 Root Cause Clustering (AI-Powered)
- **Multiple Algorithms**: K-means, DBSCAN, LDA, and HDBSCAN clustering
- **Pattern Recognition**: Identifies recurring issues and institutional pain points
- **Trend Analysis**: Detects emerging problems before they escalate
- **Visual Analytics**: Provides comprehensive reports and visualizations
- **Impact**: Enables preventive action by identifying common problems

### 5. 🤖 Emotion-Aware Chatbot
- **Emotional Intelligence**: Responds empathetically based on detected emotions
- **Escalation Triggers**: Automatically escalates based on emotional distress
- **Context Awareness**: Maintains conversation history and user emotional state
- **Multi-Language Support**: Handles various communication styles and patterns
- **Impact**: Reduces pressure on human agents and improves first-contact resolution

### 6. 🎨 Resolution Quality Predictor
- **Response Analysis**: Evaluates response effectiveness before sending
- **Quality Metrics**: Assesses tone, completeness, clarity, empathy, and actionability
- **ML-Powered**: Uses historical satisfaction data for predictions
- **Improvement Suggestions**: Provides specific recommendations for better responses
- **Impact**: Increases user satisfaction by ensuring high-quality responses

### 7. 🕵️ Anonymous Complaint with Trust Index
- **Privacy Protection**: Allows anonymous submissions while maintaining security
- **Spam Filtering**: AI-powered trust scoring to filter out spam
- **Behavioral Analysis**: Tracks patterns without compromising anonymity
- **Trust Scoring**: Multi-factor trust assessment for submission validation
- **Impact**: Balances user privacy with system protection

### 8. 📈 Organizational Mood Tracker
- **Department Analytics**: Visual dashboards showing emotional trends
- **Mood Metrics**: Tracks satisfaction, stress, engagement, and trust levels
- **Trend Detection**: Identifies mood changes over time
- **Alert System**: Notifies management of concerning trends
- **Impact**: Helps management identify stressful periods and improve wellness

### 9. 📊 Heatmap & Trend Analytics Dashboard
- **Real-Time Visualization**: Dynamic complaint heatmaps by category and department
- **Interactive Filters**: Customizable views with multiple filter options
- **Trend Charts**: Historical analysis with predictive insights
- **Performance Metrics**: Comprehensive KPI tracking and reporting
- **Impact**: Provides actionable insights for strategic decision-making

### 10. 🔄 Sentiment Shift Tracker
- **Longitudinal Analysis**: Tracks user sentiment changes over time
- **Auto-Escalation**: Triggers escalation when sentiment deteriorates
- **Pattern Recognition**: Identifies users at risk of frustration
- **Preventive Action**: Enables proactive intervention
- **Impact**: Prevents overlooked escalation and helps retain user trust

### 11. ⏰ Complaint Delay Prediction
- **Predictive Analytics**: Forecasts potential delays based on historical data
- **Workload Analysis**: Considers admin capacity and category-wise timelines
- **Early Warning**: Alerts administrators before delays occur
- **Resource Planning**: Helps optimize staff allocation
- **Impact**: Improves response times and user satisfaction

### 12. 🤖 Adaptive Machine Learning Prioritization
- **Continuous Learning**: Improves predictions based on resolution outcomes
- **Feedback Integration**: Learns from user satisfaction and admin feedback
- **Dynamic Adjustment**: Automatically refines prioritization algorithms
- **Performance Tracking**: Monitors and improves accuracy over time
- **Impact**: Makes the system smarter without manual intervention

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- 4GB+ RAM recommended
- Internet connection for AI model downloads

### Installation Steps

1. **Clone or Download the Repository**
   ```bash
   git clone <repository-url>
   cd grievance-system
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK Data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('vader_lexicon')
   ```

4. **Initialize Database**
   ```bash
   python -c "from database import DatabaseManager; DatabaseManager()"
   ```

5. **Run the Application**
   ```bash
   streamlit run enhanced_grievance_app.py
   ```

## 🚀 Usage Guide

### For Students/Users

1. **Registration/Login**
   - Create an account or login with existing credentials
   - Students can register with 'student' role

2. **Submit Grievances**
   - Use the "Submit Grievance" page
   - Choose between regular or anonymous submission
   - Attach supporting documents if needed
   - AI will analyze and route your complaint automatically

3. **Track Progress**
   - View your grievances in "My Grievances"
   - See AI analysis results and priority levels
   - Track sentiment changes over time
   - Rate resolved grievances

4. **AI Chat Assistant**
   - Get instant help through the emotion-aware chatbot
   - Ask questions about your grievances
   - Receive empathetic responses and guidance

### For Administrators/Staff

1. **Admin Dashboard**
   - Monitor overall system health
   - View key performance indicators
   - Track resolution rates and user satisfaction

2. **Manage Grievances**
   - Process complaints with AI-powered insights
   - Use quality predictor for response optimization
   - Access smart routing recommendations

3. **Analytics & Insights**
   - View comprehensive analytics dashboard
   - Monitor organizational mood trends
   - Access root cause analysis reports
   - Track department-wise performance

4. **System Configuration**
   - Adjust AI model parameters
   - Configure routing rules
   - Set up notification preferences
   - Export data for external analysis

## 🏗️ System Architecture

### Core Components

```
enhanced_grievance_app.py          # Main Streamlit application
├── advanced_sentiment_analyzer.py # AI emotion analysis
├── smart_routing_system.py        # Intelligent routing
├── root_cause_analyzer.py         # Pattern detection
├── emotion_aware_chatbot.py       # AI assistant
├── resolution_quality_predictor.py # Response optimization
├── anonymous_trust_system.py      # Privacy & security
├── mood_tracker.py               # Organizational insights
├── database.py                   # Data management
└── sentiment_analyzer.py         # Basic sentiment analysis
```

### Database Schema

The system uses SQLite with the following key tables:
- `users` - User authentication and profiles
- `grievances` - Main grievance records
- `anonymous_submissions` - Anonymous complaints with trust scores
- `chatbot_conversations` - AI chat interactions
- `mood_snapshots` - Organizational mood data
- `routing_history` - Smart routing decisions
- `root_causes` - Identified pattern clusters

## 🔧 Configuration

### AI Model Configuration

1. **Sentiment Analysis Models**
   - BERT: `j-hartmann/emotion-english-distilroberta-base`
   - RoBERTa: `cardiffnlp/twitter-roberta-base-sentiment-latest`
   - Sarcasm: `cardiffnlp/twitter-roberta-base-irony`

2. **Trust System Thresholds**
   ```python
   spam_thresholds = {
       'min_trust_score': 0.3,
       'max_submissions_per_hour': 5,
       'similarity_threshold': 0.85
   }
   ```

3. **Routing Weights**
   ```python
   routing_weights = {
       'keyword_match': 0.3,
       'category_match': 0.25,
       'staff_performance': 0.2,
       'workload_balance': 0.15,
       'specialization_match': 0.1
   }
   ```

### Performance Optimization

1. **GPU Acceleration**
   - Install CUDA for faster AI processing
   - Models automatically use GPU if available

2. **Memory Management**
   - Adjust batch sizes in model configuration
   - Monitor memory usage with large datasets

3. **Database Optimization**
   - Regular database maintenance
   - Index optimization for better query performance

## 📊 Analytics & Reporting

### Built-in Reports

1. **Executive Dashboard**
   - High-level KPIs and trends
   - Department performance overview
   - Alert notifications

2. **Mood Analytics**
   - Organizational emotional health
   - Department-wise mood trends
   - Stress and satisfaction metrics

3. **Root Cause Analysis**
   - Cluster analysis results
   - Recurring problem identification
   - Preventive action recommendations

4. **Quality Metrics**
   - Response quality scores
   - User satisfaction trends
   - Resolution effectiveness

### Export Options

- CSV export for external analysis
- JSON API for integration
- PDF reports for presentations
- Excel dashboards for stakeholders

## 🔐 Security & Privacy

### Privacy Protection
- Anonymous submissions with privacy preservation
- Encrypted sensitive data storage
- User consent management
- GDPR compliance features

### Security Measures
- Secure authentication system
- SQL injection prevention
- Input validation and sanitization
- Rate limiting for API endpoints

### Trust & Safety
- AI-powered spam detection
- Content authenticity verification
- Behavioral pattern analysis
- Abuse prevention mechanisms

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Clear transformers cache
   rm -rf ~/.cache/huggingface/transformers/
   # Reinstall dependencies
   pip install --upgrade transformers torch
   ```

2. **Database Connection Issues**
   ```python
   # Reset database
   import os
   os.remove('grievance_system.db')
   from database import DatabaseManager
   DatabaseManager()
   ```

3. **Memory Issues**
   - Reduce batch sizes in config
   - Disable GPU acceleration if needed
   - Close unnecessary browser tabs

### Performance Issues

1. **Slow AI Processing**
   - Enable GPU acceleration
   - Reduce model complexity
   - Batch process requests

2. **Database Performance**
   - Add database indexes
   - Archive old data
   - Optimize queries

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install development dependencies
4. Run tests before submitting

### Code Standards
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Document configuration changes

## 📞 Support

### Getting Help
- Check the troubleshooting section
- Review configuration options
- Submit bug reports with detailed logs
- Request features through issues

### Documentation
- API documentation available in `/docs`
- Model specifications in `/models`
- Database schema in `/database`

## 🎯 Future Enhancements

### Planned Features
- Multi-language support expansion
- Advanced ML model customization
- Real-time notification system
- Mobile application support
- Integration with external systems

### Roadmap
- Q1: Enhanced mobile interface
- Q2: Advanced analytics features
- Q3: Integration capabilities
- Q4: Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Streamlit for the web framework
- scikit-learn for machine learning utilities
- NLTK for natural language processing
- Plotly for interactive visualizations

---

**Version**: 2.0.0  
**Last Updated**: January 2025  
**Python**: 3.8+  
**Dependencies**: See requirements.txt

For questions, suggestions, or issues, please contact the development team.
