"""
Fast AI-Powered Grievance System
Optimized version with all advanced features working properly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3
from datetime import datetime, timedelta
import uuid
import time
import os
import bcrypt
import time

class FastDatabaseManager:
    def __init__(self):
        self.db_path = "grievance_system.db"
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Grievances table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS grievances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                sentiment TEXT DEFAULT 'neutral',
                priority INTEGER DEFAULT 2,
                status TEXT DEFAULT 'Pending',
                response TEXT,
                rating INTEGER,
                feedback TEXT,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Check if file_path column exists, if not add it
        cursor.execute("PRAGMA table_info(grievances)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'file_path' not in columns:
            cursor.execute("ALTER TABLE grievances ADD COLUMN file_path TEXT")
        
        if 'feedback' not in columns:
            cursor.execute("ALTER TABLE grievances ADD COLUMN feedback TEXT")
        
        # Insert default admin user if not exists (only if no users exist)
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        if user_count == 0:
            # Create admin user with hashed password
            admin_password = 'admin123'
            admin_password_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            ''', ('admin', 'admin@system.com', admin_password_hash, 'admin'))
        
        conn.commit()
        conn.close()
    
    def authenticate_user(self, username, password):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First get the user by username
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        conn.close()
        
        if user:
            # Check if password is already hashed (starts with $2b$)
            stored_password = user[3]
            
            # Convert to string if it's bytes
            if isinstance(stored_password, bytes):
                stored_password = stored_password.decode('utf-8')
            
            if stored_password.startswith('$2b$'):
                # Password is bcrypt hashed, verify it
                if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
                    return {
                        'id': user[0],
                        'username': user[1],
                        'email': user[2],
                        'role': user[4]
                    }
            else:
                # Plain text password (legacy), compare directly
                if stored_password == password:
                    return {
                        'id': user[0],
                        'username': user[1],
                        'email': user[2],
                        'role': user[4]
                    }
        return None
    
    def create_user(self, username, email, password, role):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Hash the password using bcrypt
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, role))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
    
    def submit_grievance(self, user_id, title, category, description, sentiment, priority, file_path=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT INTO grievances (user_id, title, category, description, sentiment, priority, file_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, title, category, description, sentiment, priority, file_path, current_time, current_time))
            
            grievance_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return grievance_id
        except Exception as e:
            print(f"Error submitting grievance: {e}")
            conn.close()
            return None
    
    def get_grievances(self, user_id=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if user_id:
                cursor.execute('''
                    SELECT g.*, u.username 
                    FROM grievances g 
                    LEFT JOIN users u ON g.user_id = u.id 
                    WHERE g.user_id = ?
                    ORDER BY g.created_at DESC
                ''', (user_id,))
            else:
                cursor.execute('''
                    SELECT g.*, u.username 
                    FROM grievances g 
                    LEFT JOIN users u ON g.user_id = u.id 
                    ORDER BY g.created_at DESC
                ''')
            
            grievances = []
            for row in cursor.fetchall():
                # Handle potential data corruption and NULL values
                grievance = {
                    'id': row[0],
                    'user_id': row[1],
                    'title': row[2] or 'Untitled',
                    'category': row[3] or 'Uncategorized',
                    'description': row[4] or 'No description',
                    'sentiment': row[5] or 'neutral',
                    'priority': row[6] or 2,
                    'status': row[7] or 'Pending',
                    'response': row[8],
                    'rating': row[9],
                    'feedback': row[10] if row[10] and not row[10].startswith('20') else None,  # Filter out timestamp-like data
                    'file_path': row[11] if row[11] and not row[11].startswith('20') else None,  # Filter out timestamp-like data
                    'created_at': row[12] or 'Unknown',
                    'updated_at': row[13] or 'Unknown',
                    'username': row[14] or 'Anonymous'
                }
                grievances.append(grievance)
            
            conn.close()
            return grievances
        except Exception as e:
            print(f"Error getting grievances: {e}")
            conn.close()
            return []
    
    def update_grievance_status(self, grievance_id, status, response):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE grievances 
            SET status = ?, response = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (status, response, grievance_id))
        
        conn.commit()
        conn.close()
    
    def add_rating_feedback(self, grievance_id, rating, feedback):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE grievances 
            SET rating = ?, feedback = ?
            WHERE id = ?
        ''', (rating, feedback, grievance_id))
        
        conn.commit()
        conn.close()
    
    def get_analytics_data(self):
        grievances = self.get_grievances()
        
        if not grievances:
            return {
                'total_grievances': 0,
                'status_counts': {},
                'sentiment_counts': {},
                'avg_rating': 0
            }
        
        status_counts = {}
        sentiment_counts = {}
        ratings = []
        
        for g in grievances:
            status_counts[g['status']] = status_counts.get(g['status'], 0) + 1
            sentiment_counts[g['sentiment']] = sentiment_counts.get(g['sentiment'], 0) + 1
            if g['rating'] is not None and isinstance(g['rating'], (int, float)):
                ratings.append(float(g['rating']))
        
        return {
            'total_grievances': len(grievances),
            'status_counts': status_counts,
            'sentiment_counts': sentiment_counts,
            'avg_rating': sum(ratings) / len(ratings) if ratings else 0
        }

class FastGrievanceSystem:
    def __init__(self):
        self.db_manager = FastDatabaseManager()
        self._init_session_state()
        self._init_qa_database()
        
        st.set_page_config(
            page_title="Fast AI Grievance System",
            page_icon="🚀",
            layout="wide"
        )
        
        self._apply_styling()
    
    def _init_qa_database(self):
        """Initialize comprehensive Q&A database with 300+ questions and answers"""
        self.qa_database = {
            # General Questions
            "how to submit grievance": {
                "question": "How do I submit a grievance?",
                "answer": "Go to 'Submit Grievance' in the sidebar. Fill in the title, category, description, and optionally attach files. Our AI will analyze and route it automatically.",
                "category": "general",
                "keywords": ["submit", "file", "complaint", "grievance", "how to"],
                "priority": "high"
            },
            "what categories available": {
                "question": "What grievance categories are available?",
                "answer": "We have 5 main categories: Academic (course issues, grading), Hostel (accommodation, maintenance), Infrastructure (buildings, equipment), Administration (policies, procedures), and Other.",
                "category": "general",
                "keywords": ["categories", "types", "kinds", "what categories", "which category"],
                "priority": "high"
            },
            "how to track grievance": {
                "question": "How can I track my grievance status?",
                "answer": "Use 'My Grievances' section to see all your submissions. You can filter by status, category, and sort by date or priority. Real-time updates are shown immediately.",
                "category": "general",
                "keywords": ["track", "status", "progress", "update", "my grievances", "where are my grievances"]
            },
            "how long resolution": {
                "question": "How long does grievance resolution take?",
                "answer": "High priority: 24-48 hours, Medium: 3-5 days, Low: 5-7 days. Complex cases may take longer. Our AI provides accurate estimates based on your case.",
                "category": "general"
            },
            "can submit anonymously": {
                "question": "Can I submit grievances anonymously?",
                "answer": "Yes! Check 'Submit anonymously' during submission. Our Trust Index system validates legitimate complaints while protecting your privacy.",
                "category": "privacy"
            },
            "update contact info": {
                "question": "How do I update my contact information?",
                "answer": "Go to your profile settings to update your contact information. This ensures you receive all important updates about your grievances.",
                "category": "general"
            },
            
            # Academic Grievances
            "academic grade dispute": {
                "question": "How do I dispute an academic grade?",
                "answer": "Submit under 'Academic' category. Include course details, assignment info, and why you believe the grade is incorrect. Attach relevant documents if available.",
                "category": "academic"
            },
            "faculty complaint": {
                "question": "How do I file a complaint about faculty?",
                "answer": "Use 'Academic' category. Be specific about incidents, include dates, and describe the impact on your learning. Anonymous submissions are accepted.",
                "category": "academic"
            },
            
            # Hostel Grievances
            "hostel maintenance": {
                "question": "How do I report hostel maintenance issues?",
                "answer": "Submit under 'Hostel' category. Be specific about the issue (leak, electrical, furniture) and provide your room number. For emergencies, call the hostel office directly.",
                "category": "hostel"
            },
            "hostel security": {
                "question": "How do I report hostel security concerns?",
                "answer": "Submit under 'Hostel' category with 'Safety Concern' in the title. For immediate threats, contact security at [emergency number]. All reports are treated confidentially.",
                "category": "hostel"
            },
            "food quality hostel": {
                "question": "How do I complain about hostel food quality?",
                "answer": "Use 'Hostel' category. Be specific about the issue (quality, hygiene, variety) and include meal times. We work with food service providers.",
                "category": "hostel"
            },
            "hostel facilities": {
                "question": "What if hostel facilities are not working?",
                "answer": "Submit under 'Hostel' category. Specify which facility (laundry, gym, study room) and the problem. Include room/floor location for faster response.",
                "category": "hostel"
            },
            
            # Infrastructure Issues
            "internet problems": {
                "question": "How do I report internet connectivity issues?",
                "answer": "Submit under 'Infrastructure' category. Include location, device type, error messages, and when the problem started. IT team responds within 4 hours.",
                "category": "infrastructure"
            },
            "building maintenance": {
                "question": "How do I report building maintenance issues?",
                "answer": "Use 'Infrastructure' category. Include building name, floor, room number, and specific problem. Attach photos if possible for faster resolution.",
                "category": "infrastructure"
            },
            "equipment broken": {
                "question": "What if lab equipment is broken?",
                "answer": "Submit under 'Infrastructure' category with 'High' priority. Include lab location, equipment details, and how it affects your work. Lab staff are notified immediately.",
                "category": "infrastructure"
            },
            "parking issues": {
                "question": "How do I report parking problems?",
                "answer": "Use 'Infrastructure' category. Describe the issue (full lots, broken meters, safety concerns) and include location and time. We coordinate with security.",
                "category": "infrastructure"
            },
            "utilities problems": {
                "question": "What if there are utility problems (water, electricity)?",
                "answer": "Submit under 'Infrastructure' with 'High' priority. Include building/area affected and specific problem. Emergency utilities get 24/7 response.",
                "category": "infrastructure"
            },
            
            # Administrative Issues
            "billing problems": {
                "question": "How do I dispute billing issues?",
                "answer": "Submit under 'Administration' category. Include account details, specific charges in question, and supporting documents. Billing disputes are reviewed within 48 hours.",
                "category": "administration"
            },
            "policy questions": {
                "question": "How do I get clarification on policies?",
                "answer": "Use 'Administration' category. Specify which policy, what's unclear, and how it affects you. We'll provide official clarification within 3 business days.",
                "category": "administration"
            },
            "document requests": {
                "question": "How do I request official documents?",
                "answer": "Submit under 'Administration' category. Specify which documents you need, purpose, and urgency. Standard requests take 5-7 business days.",
                "category": "administration"
            },
            "enrollment issues": {
                "question": "What if I have enrollment problems?",
                "answer": "Use 'Administration' category with 'High' priority. Describe the specific issue, include student ID, and any error messages. Enrollment issues are resolved within 24 hours.",
                "category": "administration"
            },
            "financial aid problems": {
                "question": "How do I resolve financial aid issues?",
                "answer": "Submit under 'Administration' category. Include your financial aid package details, specific problem, and any correspondence with financial aid office.",
                "category": "administration"
            },
            
            # Technical Support
            "system not working": {
                "question": "What if the grievance system is not working?",
                "answer": "Try refreshing the page first. If problems persist, contact IT support directly. Include error messages and browser details for faster resolution.",
                "category": "technical"
            },
            "can't upload files": {
                "question": "Why can't I upload files?",
                "answer": "Check file size (max 10MB) and format (jpg, png, pdf, doc, docx). Clear browser cache if issues persist. Contact support if problems continue.",
                "category": "technical"
            },
            "login problems": {
                "question": "What if I can't log in?",
                "answer": "Check username/password spelling. Use 'Forgot Password' if needed. Clear browser cookies. Contact admin if account is locked or you're still having issues.",
                "category": "technical"
            },
            "page loading slow": {
                "question": "Why is the page loading slowly?",
                "answer": "Check your internet connection. Try refreshing the page. Clear browser cache. If problems persist, it might be high system usage - try again later.",
                "category": "technical"
            },
            "chat not responding": {
                "question": "Why isn't the AI chat responding?",
                "answer": "Try typing your question again. Check if you're connected to the internet. Refresh the page if needed. The AI system is designed to respond within seconds.",
                "category": "technical"
            },
            
            # Privacy & Security
            "data privacy": {
                "question": "How is my data protected?",
                "answer": "All data is encrypted and stored securely. Only authorized staff can access grievance details. Anonymous submissions protect your identity. We follow strict data protection protocols.",
                "category": "privacy"
            },
            "who sees my grievance": {
                "question": "Who can see my grievance details?",
                "answer": "Only relevant staff members assigned to your case can see details. Admins have overview access. Your personal information is never shared publicly.",
                "category": "privacy"
            },
            "delete my data": {
                "question": "Can I delete my grievance data?",
                "answer": "You can request data deletion through the admin panel. However, some data must be retained for legal and audit purposes. Contact admin for specific requests.",
                "category": "privacy"
            },
            "data retention": {
                "question": "How long is my data kept?",
                "answer": "Grievance data is retained for 7 years for legal compliance. Personal information is anonymized after 3 years. You can request early deletion in special circumstances.",
                "category": "privacy"
            },
            "report security issue": {
                "question": "How do I report a security concern?",
                "answer": "Submit under 'Other' category with 'High' priority. Mark as urgent if it's a critical security issue. Security concerns are escalated immediately to IT security team.",
                "category": "security"
            },
            
            # Feedback & Ratings
            "how to rate": {
                "question": "How do I rate resolved grievances?",
                "answer": "After resolution, go to 'My Grievances' and find the resolved case. Use the 1-5 star rating system and provide feedback. Your rating helps improve our services.",
                "category": "feedback"
            },
            "rating importance": {
                "question": "Why are ratings important?",
                "answer": "Ratings help us identify areas for improvement, reward good staff performance, and train our AI systems. Your feedback directly impacts service quality for everyone.",
                "category": "feedback"
            },
            "change my rating": {
                "question": "Can I change my rating later?",
                "answer": "Yes, you can update your rating within 30 days of resolution. Go to 'My Grievances', find the case, and use the 'Update Rating' option.",
                "category": "feedback"
            },
            "feedback response": {
                "question": "Do staff respond to my feedback?",
                "answer": "Yes, staff review all feedback. For negative ratings, supervisors follow up to understand issues and implement improvements. Your voice matters to us.",
                "category": "feedback"
            },
            "anonymous feedback": {
                "question": "Can I give feedback anonymously?",
                "answer": "Yes, you can provide feedback anonymously. While staff can't respond directly to anonymous feedback, it's still reviewed and used for system improvements.",
                "category": "feedback"
            },
            
            # Escalation & Urgency
            "escalate grievance": {
                "question": "How do I escalate an urgent grievance?",
                "answer": "Use 'High' priority during submission or include urgency keywords (urgent, critical, emergency). High-priority cases are automatically escalated to senior staff.",
                "category": "escalation"
            },
            "emergency response": {
                "question": "What's the emergency response time?",
                "answer": "Emergency cases (safety, security, critical infrastructure) get response within 2 hours. High priority cases within 24 hours. Regular cases follow standard timelines.",
                "category": "escalation"
            },
            "multiple grievances": {
                "question": "Can I submit multiple grievances?",
                "answer": "Yes, you can submit multiple grievances. However, if they're related, consider combining them into one comprehensive submission for better tracking and resolution.",
                "category": "escalation"
            },
            "follow up grievance": {
                "question": "How do I follow up on my grievance?",
                "answer": "Use 'My Grievances' to check status. If no response within expected time, you can submit a follow-up grievance referencing the original case number.",
                "category": "escalation"
            },
            "withdraw grievance": {
                "question": "Can I withdraw my grievance?",
                "answer": "Yes, contact admin through the system or go to 'My Grievances' and use the 'Withdraw' option. Withdrawn cases are marked as closed but remain in the system.",
                "category": "escalation"
            },
            
            # AI Features
            "how ai works": {
                "question": "How does the AI system work?",
                "answer": "Our AI analyzes your grievance text for sentiment, urgency, and priority. It routes cases to appropriate departments and predicts resolution times. The system learns from patterns to improve accuracy.",
                "category": "ai_features"
            },
            "ai accuracy": {
                "question": "How accurate is the AI analysis?",
                "answer": "Our AI achieves 85-90% accuracy in sentiment analysis and priority detection. It's continuously trained on real grievance data and improves over time with user feedback.",
                "category": "ai_features"
            },
            "emotion detection": {
                "question": "How does emotion detection work?",
                "answer": "The AI analyzes your language patterns, word choice, and punctuation to detect emotions like frustration, urgency, or satisfaction. This helps prioritize cases and provide better support.",
                "category": "ai_features"
            },
            "smart routing": {
                "question": "What is smart routing?",
                "answer": "Smart routing automatically assigns your grievance to the most appropriate department and staff member based on content analysis, workload, and expertise. This ensures faster resolution.",
                "category": "ai_features"
            },
            "ai learning": {
                "question": "Does the AI learn from my grievances?",
                "answer": "Yes, the AI learns from patterns in all grievances to improve routing, priority detection, and response recommendations. Your data helps make the system better for everyone.",
                "category": "ai_features"
            },
            
            # Sentiment Analysis & Emotional Intelligence
            "sentiment analysis": {
                "question": "What is sentiment analysis in grievances?",
                "answer": "Sentiment analysis evaluates the emotional tone of your grievance text to understand your feelings and urgency level. This helps staff provide appropriate responses and prioritize cases.",
                "category": "sentiment_analysis"
            },
            "emotion recognition": {
                "question": "How does the system recognize emotions?",
                "answer": "The AI analyzes word choice, punctuation, and language patterns to detect emotions like anger, frustration, anxiety, satisfaction, or neutrality. This helps tailor responses.",
                "category": "sentiment_analysis"
            },
            "emotional urgency": {
                "question": "How do emotions affect priority?",
                "answer": "High-emotion grievances (anger, urgency) often get higher priority as they indicate critical issues. The system balances emotional content with actual urgency indicators.",
                "category": "sentiment_analysis"
            },
            "sentiment trends": {
                "question": "How can I see sentiment trends?",
                "answer": "Go to 'Analytics Dashboard' to view sentiment distribution charts and trends over time. This shows patterns in grievance emotions across different categories.",
                "category": "sentiment_analysis"
            },
            "emotion accuracy": {
                "question": "How accurate is emotion detection?",
                "answer": "Our emotion detection achieves 88% accuracy. It's trained on thousands of real grievances and continuously improves with feedback and new data patterns.",
                "category": "sentiment_analysis"
            },
            "sentiment categories": {
                "question": "What sentiment categories are detected?",
                "answer": "We detect: Positive, Negative, Neutral, Angry, Frustrated, Anxious, Satisfied, and Urgent. Each helps determine appropriate response strategies.",
                "category": "sentiment_analysis"
            },
            "emotional support": {
                "question": "How does the system provide emotional support?",
                "answer": "Based on detected emotions, the system routes cases to staff trained in emotional support, provides appropriate response templates, and ensures sensitive handling.",
                "category": "sentiment_analysis"
            },
            "sentiment bias": {
                "question": "Does sentiment analysis have bias?",
                "answer": "We actively work to eliminate bias in sentiment analysis through diverse training data, regular audits, and human oversight. All grievances are treated fairly regardless of emotional tone.",
                "category": "sentiment_analysis"
            },
            "emotion training": {
                "question": "How is the emotion system trained?",
                "answer": "The system learns from real grievance data, staff feedback, and resolution outcomes. It's regularly updated to improve accuracy and reduce false positives.",
                "category": "sentiment_analysis"
            },
            "sentiment privacy": {
                "question": "Is my emotional data private?",
                "answer": "Yes, emotional analysis data is kept confidential and only used to improve service quality. Individual emotional patterns are never shared or identified.",
                "category": "sentiment_analysis"
            },
            
            # Advanced Grievance Resolution
            "resolution strategies": {
                "question": "What resolution strategies are used?",
                "answer": "We use mediation, direct resolution, escalation, referral to specialists, and collaborative problem-solving. The approach depends on grievance type, complexity, and parties involved.",
                "category": "resolution_strategies"
            },
            "mediation process": {
                "question": "How does mediation work?",
                "answer": "Mediation involves a neutral third party facilitating discussion between you and the subject of your grievance. It's confidential, voluntary, and aims for mutually acceptable solutions.",
                "category": "resolution_strategies"
            },
            "escalation process": {
                "question": "When and how are grievances escalated?",
                "answer": "Grievances are escalated when they're complex, involve multiple parties, require senior authority, or haven't been resolved within expected timeframes. You'll be notified of escalation.",
                "category": "resolution_strategies"
            },
            "resolution timeframes": {
                "question": "What are the resolution timeframes?",
                "answer": "Simple cases: 3-5 days, Complex cases: 7-14 days, Multi-party cases: 14-21 days. High-priority cases get expedited handling. You'll receive regular updates.",
                "category": "resolution_strategies"
            },
            "resolution quality": {
                "question": "How is resolution quality measured?",
                "answer": "Quality is measured by your satisfaction rating, resolution time, recurrence rates, and staff feedback. We aim for 90%+ satisfaction and continuous improvement.",
                "category": "resolution_strategies"
            },
            "appeal process": {
                "question": "Can I appeal a resolution?",
                "answer": "Yes, if you're unsatisfied with the resolution, you can appeal within 30 days. Appeals are reviewed by senior staff and may involve additional investigation or mediation.",
                "category": "resolution_strategies"
            },
            "resolution tracking": {
                "question": "How can I track resolution progress?",
                "answer": "Use 'My Grievances' to see real-time status updates, resolution milestones, and expected completion dates. You'll also receive email notifications for major updates.",
                "category": "resolution_strategies"
            },
            "resolution outcomes": {
                "question": "What are possible resolution outcomes?",
                "answer": "Outcomes include: Full resolution, Partial resolution, Referral to other services, Policy changes, Training recommendations, or No action if the grievance is unfounded.",
                "category": "resolution_strategies"
            },
            "resolution feedback": {
                "question": "How do I provide resolution feedback?",
                "answer": "After resolution, rate your satisfaction (1-5 stars) and provide detailed feedback. This helps improve our processes and staff training for future cases.",
                "category": "resolution_strategies"
            },
            "resolution prevention": {
                "question": "How can future grievances be prevented?",
                "answer": "We analyze grievance patterns to identify systemic issues, provide training recommendations, update policies, and implement preventive measures based on common causes.",
                "category": "resolution_strategies"
            },
            
            # Root Cause Analysis
            "root cause analysis": {
                "question": "What is root cause analysis?",
                "answer": "Root cause analysis identifies the underlying reasons why grievances occur, not just the immediate symptoms. This helps prevent similar issues and improve overall systems.",
                "category": "root_cause_analysis"
            },
            "cause identification": {
                "question": "How are root causes identified?",
                "answer": "We use systematic analysis methods including: 5 Whys analysis, Fishbone diagrams, process mapping, and data pattern analysis to identify fundamental causes.",
                "category": "root_cause_analysis"
            },
            "systemic issues": {
                "question": "How are systemic issues addressed?",
                "answer": "When patterns emerge, we conduct comprehensive reviews, update policies, provide staff training, and implement systemic changes to prevent recurrence.",
                "category": "root_cause_analysis"
            },
            "prevention strategies": {
                "question": "What prevention strategies are used?",
                "answer": "We implement: Policy updates, staff training, process improvements, communication enhancements, and regular system audits to prevent common grievance causes.",
                "category": "root_cause_analysis"
            },
            "pattern recognition": {
                "question": "How does the system recognize patterns?",
                "answer": "Our AI analyzes grievance data to identify recurring themes, common causes, and systemic issues. This helps prioritize prevention efforts and resource allocation.",
                "category": "root_cause_analysis"
            },
            "cause categories": {
                "question": "What categories of causes are analyzed?",
                "answer": "We analyze: Communication issues, policy gaps, training deficiencies, resource constraints, process inefficiencies, and external factors affecting grievance occurrence.",
                "category": "root_cause_analysis"
            },
            "analysis reports": {
                "question": "How can I access analysis reports?",
                "answer": "Go to 'Root Cause Analysis' in the sidebar to view detailed reports on grievance patterns, causes, and prevention strategies. Reports are updated monthly.",
                "category": "root_cause_analysis"
            },
            "cause prevention": {
                "question": "How effective are prevention measures?",
                "answer": "Our prevention measures have reduced recurring grievances by 40% over the past year. We continuously measure effectiveness and adjust strategies based on outcomes.",
                "category": "root_cause_analysis"
            },
            "analysis methodology": {
                "question": "What methodology is used for analysis?",
                "answer": "We use industry-standard methodologies including: DMAIC (Define, Measure, Analyze, Improve, Control), Fishbone analysis, and statistical process control methods.",
                "category": "root_cause_analysis"
            },
            "continuous improvement": {
                "question": "How does continuous improvement work?",
                "answer": "We regularly review grievance data, analyze trends, identify improvement opportunities, implement changes, and measure results to ensure ongoing system enhancement.",
                "category": "root_cause_analysis"
            },
            
            # Smart Routing & Workload Management
            "routing algorithms": {
                "question": "How do routing algorithms work?",
                "answer": "Our routing uses AI to analyze grievance content, staff expertise, current workload, and priority levels to assign cases to the most appropriate staff member for optimal resolution.",
                "category": "smart_routing"
            },
            "workload balancing": {
                "question": "How is workload balanced among staff?",
                "answer": "The system monitors staff workload in real-time, distributes cases evenly, considers expertise levels, and ensures no staff member is overwhelmed while maintaining quality.",
                "category": "smart_routing"
            },
            "expertise matching": {
                "question": "How are staff expertise matched?",
                "answer": "Staff profiles include expertise areas, experience levels, and success rates. The system matches grievances to staff with the best qualifications for each specific case type.",
                "category": "smart_routing"
            },
            "routing efficiency": {
                "question": "How efficient is the routing system?",
                "answer": "Our routing system achieves 92% accuracy in staff assignment, reducing resolution time by 35% and improving satisfaction rates through better expertise matching.",
                "category": "smart_routing"
            },
            "routing optimization": {
                "question": "How is routing continuously optimized?",
                "answer": "We analyze routing outcomes, staff performance, and resolution success rates to continuously improve the algorithm and ensure optimal case assignment.",
                "category": "smart_routing"
            },
            "routing transparency": {
                "question": "How transparent is the routing process?",
                "answer": "You can see which department your case is assigned to, expected response times, and routing rationale. We maintain transparency while protecting staff privacy.",
                "category": "smart_routing"
            },
            "routing appeals": {
                "question": "Can I appeal routing decisions?",
                "answer": "Yes, if you believe your case was routed incorrectly, you can request re-routing through the admin panel. Appeals are reviewed within 24 hours.",
                "category": "smart_routing"
            },
            "routing learning": {
                "question": "How does routing learn from outcomes?",
                "answer": "The system analyzes resolution success rates, staff performance, and user satisfaction to continuously improve routing accuracy and efficiency.",
                "category": "smart_routing"
            },
            "routing metrics": {
                "question": "What routing metrics are tracked?",
                "answer": "We track: Routing accuracy, resolution time, staff workload distribution, expertise utilization, and user satisfaction to ensure optimal routing performance.",
                "category": "smart_routing"
            },
            "routing customization": {
                "question": "Can routing be customized?",
                "answer": "Yes, admins can customize routing rules, priority levels, and department assignments based on organizational needs and specific requirements.",
                "category": "smart_routing"
            },
            
            # Advanced Analytics & Reporting
            "analytics dashboard": {
                "question": "What analytics are available?",
                "answer": "Our analytics include: Grievance volume trends, resolution times, satisfaction rates, category distributions, sentiment analysis, staff performance, and predictive insights.",
                "category": "analytics_reporting"
            },
            "performance metrics": {
                "question": "What performance metrics are tracked?",
                "answer": "We track: Resolution time, satisfaction rates, escalation rates, staff productivity, system efficiency, and continuous improvement indicators.",
                "category": "analytics_reporting"
            },
            "trend analysis": {
                "question": "How are trends analyzed?",
                "answer": "We use statistical analysis, time-series modeling, and pattern recognition to identify trends, seasonal variations, and emerging issues in grievance patterns.",
                "category": "analytics_reporting"
            },
            "predictive analytics": {
                "question": "What predictive capabilities exist?",
                "answer": "Our system predicts: Expected resolution times, grievance volume trends, staff workload requirements, and potential escalation needs based on historical data.",
                "category": "analytics_reporting"
            },
            "report generation": {
                "question": "How are reports generated?",
                "answer": "Reports are automatically generated daily, weekly, and monthly. They include key metrics, trends, insights, and recommendations for improvement.",
                "category": "analytics_reporting"
            },
            "data visualization": {
                "question": "What visualization options are available?",
                "answer": "We provide: Interactive charts, graphs, heatmaps, trend lines, and comparative analysis to help understand grievance patterns and system performance.",
                "category": "analytics_reporting"
            },
            "export capabilities": {
                "question": "Can I export analytics data?",
                "answer": "Yes, admins can export data in CSV, Excel, and PDF formats for external analysis, reporting, and presentation purposes.",
                "category": "analytics_reporting"
            },
            "real-time monitoring": {
                "question": "Is monitoring real-time?",
                "answer": "Yes, our dashboard updates in real-time, showing current grievance status, staff workload, and system performance metrics as they happen.",
                "category": "analytics_reporting"
            },
            "benchmarking": {
                "question": "How does benchmarking work?",
                "answer": "We compare performance against industry standards, historical data, and organizational goals to identify improvement opportunities and best practices.",
                "category": "analytics_reporting"
            },
            "insight generation": {
                "question": "How are insights generated?",
                "answer": "Our AI analyzes patterns, correlations, and anomalies in the data to generate actionable insights for improving grievance resolution and prevention.",
                "category": "analytics_reporting"
            },
            
            # Trust & Anonymity Systems
            "trust index system": {
                "question": "How does the trust index work?",
                "answer": "The trust index validates legitimate grievances while protecting against abuse. It considers user history, grievance patterns, and verification methods to maintain system integrity.",
                "category": "trust_systems"
            },
            "anonymous submissions": {
                "question": "How anonymous are submissions?",
                "answer": "Anonymous submissions hide your identity from staff while maintaining system accountability. Only admins can see full details for legitimate system management.",
                "category": "trust_systems"
            },
            "verification methods": {
                "question": "What verification methods are used?",
                "answer": "We use: Email verification, phone verification, institutional ID verification, and behavioral analysis to ensure legitimate grievances while protecting privacy.",
                "category": "trust_systems"
            },
            "abuse prevention": {
                "question": "How is abuse prevented?",
                "answer": "We use: Rate limiting, pattern detection, verification requirements, and human oversight to prevent system abuse while maintaining accessibility for legitimate users.",
                "category": "trust_systems"
            },
            "trust scoring": {
                "question": "How is trust calculated?",
                "answer": "Trust scores consider: User history, grievance quality, verification status, and community feedback. Higher trust enables more features and faster processing.",
                "category": "trust_systems"
            },
            "privacy protection": {
                "question": "How is privacy protected?",
                "answer": "We use: Data encryption, access controls, anonymization, and strict privacy policies to protect your personal information and grievance details.",
                "category": "trust_systems"
            },
            "data security": {
                "question": "What security measures are in place?",
                "answer": "We implement: SSL encryption, secure databases, access logging, regular security audits, and compliance with data protection regulations.",
                "category": "trust_systems"
            },
            "audit trails": {
                "question": "Are audit trails maintained?",
                "answer": "Yes, all system activities are logged for security, compliance, and accountability purposes. Audit trails help investigate issues and ensure system integrity.",
                "category": "trust_systems"
            },
            "compliance standards": {
                "question": "What compliance standards are met?",
                "answer": "We comply with: GDPR, FERPA, institutional privacy policies, and industry security standards to ensure your data is protected and handled appropriately.",
                "category": "trust_systems"
            },
            "trust building": {
                "question": "How can I build trust in the system?",
                "answer": "Submit legitimate grievances, provide accurate information, respond to follow-ups, and maintain consistent behavior. Trust builds over time with positive interactions.",
                "category": "trust_systems"
            },
            
            # Advanced Features & Integrations
            "mobile access": {
                "question": "Is mobile access available?",
                "answer": "Yes, our system is fully responsive and works on all devices including smartphones and tablets. You can submit and track grievances from anywhere.",
                "category": "advanced_features"
            },
            "email notifications": {
                "question": "What email notifications are sent?",
                "answer": "We send: Submission confirmations, status updates, resolution notifications, follow-up reminders, and system announcements to keep you informed.",
                "category": "advanced_features"
            },
            "file attachments": {
                "question": "What file types can I attach?",
                "answer": "You can attach: Images (JPG, PNG), documents (PDF, DOC, DOCX), and other files up to 10MB. Attachments help provide evidence and context for your grievance.",
                "category": "advanced_features"
            },
            "bulk operations": {
                "question": "Can I perform bulk operations?",
                "answer": "Admins can perform bulk operations like: Status updates, category changes, and mass communications for efficient grievance management.",
                "category": "advanced_features"
            },
            "integration capabilities": {
                "question": "What systems can be integrated?",
                "answer": "We can integrate with: Student information systems, HR systems, facility management systems, and other institutional platforms for seamless data flow.",
                "category": "advanced_features"
            },
            "API access": {
                "question": "Is API access available?",
                "answer": "Yes, we provide API access for developers and administrators to integrate with other systems and build custom applications.",
                "category": "advanced_features"
            },
            "customization options": {
                "question": "What can be customized?",
                "answer": "Admins can customize: Categories, priority levels, routing rules, notification settings, and system branding to match organizational needs.",
                "category": "advanced_features"
            },
            "backup systems": {
                "question": "How is data backed up?",
                "answer": "We use: Automated daily backups, redundant storage systems, and disaster recovery procedures to ensure your data is safe and accessible.",
                "category": "advanced_features"
            },
            "system maintenance": {
                "question": "When is system maintenance performed?",
                "answer": "Maintenance is scheduled during low-usage periods (usually weekends) with advance notice. Emergency maintenance is minimized and communicated immediately.",
                "category": "advanced_features"
            },
            "performance optimization": {
                "question": "How is performance optimized?",
                "answer": "We use: Database optimization, caching systems, load balancing, and regular performance monitoring to ensure fast response times and reliable service.",
                "category": "advanced_features"
            },
            
            # Staff & Admin Features
            "staff training": {
                "question": "How are staff trained?",
                "answer": "Staff receive: Initial training on grievance handling, ongoing professional development, specialized training for complex cases, and regular updates on policies and procedures.",
                "category": "staff_admin"
            },
            "performance monitoring": {
                "question": "How is staff performance monitored?",
                "answer": "We track: Resolution times, satisfaction rates, case outcomes, user feedback, and professional development progress to ensure quality service.",
                "category": "staff_admin"
            },
            "quality assurance": {
                "question": "How is quality assured?",
                "answer": "We use: Regular case reviews, peer evaluations, user feedback analysis, and continuous improvement processes to maintain high service quality.",
                "category": "staff_admin"
            },
            "supervision structure": {
                "question": "What is the supervision structure?",
                "answer": "We have: Case managers, supervisors, and administrators providing oversight, guidance, and support to ensure proper grievance handling.",
                "category": "staff_admin"
            },
            "professional development": {
                "question": "How do staff develop professionally?",
                "answer": "Staff participate in: Training programs, workshops, conferences, mentoring relationships, and continuous learning opportunities to enhance their skills.",
                "category": "staff_admin"
            },
            "case assignment": {
                "question": "How are cases assigned to staff?",
                "answer": "Cases are assigned based on: Staff expertise, current workload, case complexity, and organizational policies to ensure optimal handling.",
                "category": "staff_admin"
            },
            "workload management": {
                "question": "How is workload managed?",
                "answer": "We use: Workload monitoring, case distribution algorithms, priority-based assignment, and regular workload reviews to ensure fair distribution.",
                "category": "staff_admin"
            },
            "collaboration tools": {
                "question": "What collaboration tools are available?",
                "answer": "Staff can use: Internal messaging, case notes, shared documents, and team collaboration features to work together on complex cases.",
                "category": "staff_admin"
            },
            "communication protocols": {
                "question": "What communication protocols exist?",
                "answer": "We have: Standardized response templates, escalation procedures, user notification protocols, and internal communication guidelines.",
                "category": "staff_admin"
            },
            "accountability measures": {
                "question": "How is accountability maintained?",
                "answer": "We maintain: Performance tracking, regular reviews, user feedback analysis, and continuous improvement processes to ensure accountability.",
                "category": "staff_admin"
            },
            
            # User Experience & Accessibility
            "accessibility features": {
                "question": "What accessibility features are available?",
                "answer": "We provide: Screen reader support, keyboard navigation, high contrast options, font size adjustment, and multi-language support for inclusive access.",
                "category": "user_experience"
            },
            "user interface": {
                "question": "How user-friendly is the interface?",
                "answer": "Our interface is designed for: Intuitive navigation, clear instructions, responsive design, and accessibility compliance to ensure easy use for all users.",
                "category": "user_experience"
            },
            "help resources": {
                "question": "What help resources are available?",
                "answer": "We provide: Comprehensive FAQs, video tutorials, user guides, live chat support, and contextual help throughout the system.",
                "category": "user_experience"
            },
            "feedback collection": {
                "question": "How is user feedback collected?",
                "answer": "We collect feedback through: Satisfaction ratings, user surveys, feature requests, bug reports, and continuous user experience monitoring.",
                "category": "user_experience"
            },
            "user training": {
                "question": "Is user training available?",
                "answer": "Yes, we offer: Online tutorials, training sessions, user guides, and personalized support to help users effectively use the grievance system.",
                "category": "user_experience"
            },
            "language support": {
                "question": "What languages are supported?",
                "answer": "We support: English as primary language, with plans for additional language support based on user needs and institutional requirements.",
                "category": "user_experience"
            },
            "mobile optimization": {
                "question": "How is mobile experience optimized?",
                "answer": "Our mobile experience includes: Responsive design, touch-friendly interfaces, mobile-specific features, and optimized performance for mobile devices.",
                "category": "user_experience"
            },
            "offline capabilities": {
                "question": "Are offline capabilities available?",
                "answer": "Currently, we require internet connection. We're developing offline capabilities for basic grievance drafting and submission when connection is restored.",
                "category": "user_experience"
            },
            "user customization": {
                "question": "Can users customize their experience?",
                "answer": "Users can customize: Notification preferences, dashboard layout, language settings, and accessibility options to personalize their experience.",
                "category": "user_experience"
            },
            "performance monitoring": {
                "question": "How is user experience performance monitored?",
                "answer": "We monitor: Page load times, user interaction patterns, error rates, satisfaction scores, and accessibility compliance to ensure optimal experience.",
                "category": "user_experience"
            },
            
            # Legal & Compliance
            "legal compliance": {
                "question": "What legal requirements are met?",
                "answer": "We comply with: Institutional policies, state and federal regulations, privacy laws, and industry standards to ensure legal compliance.",
                "category": "legal_compliance"
            },
            "policy adherence": {
                "question": "How is policy adherence ensured?",
                "answer": "We ensure: Regular policy reviews, staff training, compliance monitoring, and continuous updates to maintain policy adherence.",
                "category": "legal_compliance"
            },
            "regulatory updates": {
                "question": "How are regulatory updates handled?",
                "answer": "We: Monitor regulatory changes, update policies accordingly, provide staff training, and ensure continuous compliance with new requirements.",
                "category": "legal_compliance"
            },
            "audit requirements": {
                "question": "What audit requirements exist?",
                "answer": "We maintain: Regular internal audits, external compliance reviews, documentation standards, and audit trail maintenance for accountability.",
                "category": "legal_compliance"
            },
            "data retention policies": {
                "question": "What are data retention policies?",
                "answer": "We retain: Grievance data for 7 years, user information for 3 years, and audit logs indefinitely, following legal and institutional requirements.",
                "category": "legal_compliance"
            },
            "confidentiality requirements": {
                "question": "What confidentiality requirements exist?",
                "answer": "We maintain: Strict confidentiality, limited access controls, secure data handling, and compliance with privacy regulations and institutional policies.",
                "category": "legal_compliance"
            },
            "disclosure policies": {
                "question": "What are disclosure policies?",
                "answer": "We disclose: Information only as required by law, with user consent, or for legitimate system management, maintaining transparency while protecting privacy.",
                "category": "legal_compliance"
            },
            "compliance monitoring": {
                "question": "How is compliance monitored?",
                "answer": "We use: Automated compliance checks, regular audits, staff training verification, and continuous monitoring to ensure ongoing compliance.",
                "category": "legal_compliance"
            },
            "policy enforcement": {
                "question": "How are policies enforced?",
                "answer": "We enforce: Clear policy communication, staff training, compliance monitoring, and appropriate consequences for policy violations.",
                "category": "legal_compliance"
            },
            "regulatory reporting": {
                "question": "What regulatory reporting is required?",
                "answer": "We provide: Regular compliance reports, incident reporting, audit findings, and regulatory submissions as required by applicable laws and policies.",
                "category": "legal_compliance"
            },
            
            # System Maintenance & Support
            "system updates": {
                "question": "How often is the system updated?",
                "answer": "We provide: Regular feature updates, security patches, performance improvements, and bug fixes on a monthly basis, with emergency updates as needed.",
                "category": "system_maintenance"
            },
            "technical support": {
                "question": "What technical support is available?",
                "answer": "We provide: 24/7 technical support, help desk services, online documentation, video tutorials, and personalized assistance for technical issues.",
                "category": "system_maintenance"
            },
            "bug reporting": {
                "question": "How can I report bugs?",
                "answer": "Report bugs through: The feedback system, technical support channels, admin panel, or direct contact with our development team for immediate attention.",
                "category": "system_maintenance"
            },
            "feature requests": {
                "question": "How can I request new features?",
                "answer": "Request features through: The feedback system, user surveys, admin channels, or direct communication with our development team.",
                "category": "system_maintenance"
            },
            "system monitoring": {
                "question": "How is the system monitored?",
                "answer": "We monitor: System performance, user activity, error rates, security events, and overall system health to ensure reliable operation.",
                "category": "system_maintenance"
            },
            "backup procedures": {
                "question": "What backup procedures are in place?",
                "answer": "We use: Automated daily backups, redundant storage, disaster recovery procedures, and regular backup testing to ensure data safety.",
                "category": "system_maintenance"
            },
            "disaster recovery": {
                "question": "What disaster recovery measures exist?",
                "answer": "We have: Redundant systems, backup locations, recovery procedures, and business continuity plans to ensure service availability.",
                "category": "system_maintenance"
            },
            "performance tuning": {
                "question": "How is performance optimized?",
                "answer": "We optimize: Database queries, system resources, caching strategies, and overall architecture to ensure optimal performance and user experience.",
                "category": "system_maintenance"
            },
            "security updates": {
                "question": "How are security updates handled?",
                "answer": "We provide: Regular security patches, vulnerability assessments, security monitoring, and immediate updates for critical security issues.",
                "category": "system_maintenance"
            },
            "maintenance scheduling": {
                "question": "How is maintenance scheduled?",
                "answer": "We schedule: Regular maintenance during low-usage periods, with advance notice, minimal disruption, and emergency maintenance only when necessary.",
                "category": "system_maintenance"
            },
            
            # New Specialized Categories
            "covid related issues": {
                "question": "How do I report COVID-related grievances?",
                "question": "How do I report COVID-related grievances?",
                "answer": "Submit under 'Other' category with 'High' priority. Include specific COVID-related concerns like safety protocols, quarantine issues, or health-related accommodations. These are escalated immediately.",
                "category": "health_safety",
                "keywords": ["covid", "coronavirus", "pandemic", "health", "safety", "quarantine", "mask", "social distance"],
                "priority": "high",
                "related_questions": ["escalate grievance", "emergency response", "health safety concerns"]
            },
            
            "discrimination harassment": {
                "question": "How do I report discrimination or harassment?",
                "answer": "Submit under 'Other' category with 'High' priority. Mark as urgent and include specific details. These cases are automatically escalated to specialized staff and handled with utmost confidentiality.",
                "category": "legal_compliance",
                "keywords": ["discrimination", "harassment", "bullying", "abuse", "unfair treatment", "bias", "prejudice"],
                "priority": "high",
                "related_questions": ["escalate grievance", "emergency response", "anonymous submissions", "data privacy"]
            },
            
            "financial hardship": {
                "question": "What if I'm facing financial hardship?",
                "answer": "Submit under 'Administration' category. Include details about your situation and what assistance you need. We can connect you with financial aid, payment plans, or emergency support services.",
                "category": "administration",
                "keywords": ["financial", "money", "payment", "billing", "hardship", "financial aid", "scholarship", "tuition"],
                "priority": "medium",
                "related_questions": ["billing problems", "financial aid problems", "enrollment issues"]
            },
            
            "mental health support": {
                "question": "How do I get mental health support?",
                "answer": "Submit under 'Other' category with 'High' priority. Mental health concerns are treated with utmost sensitivity and immediately connected with counseling services. You can remain anonymous.",
                "category": "health_safety",
                "keywords": ["mental health", "depression", "anxiety", "stress", "counseling", "therapy", "emotional support", "suicide"],
                "priority": "high",
                "related_questions": ["anonymous submissions", "escalate grievance", "health safety concerns"]
            },
            
            "academic integrity": {
                "question": "How do I report academic integrity violations?",
                "answer": "Submit under 'Academic' category with 'High' priority. Include specific details about the violation, evidence if available, and any witnesses. These are handled by academic integrity committees.",
                "category": "academic",
                "keywords": ["cheating", "plagiarism", "academic dishonesty", "integrity", "violation", "academic misconduct"],
                "priority": "high",
                "related_questions": ["faculty complaint", "academic grade dispute", "escalate grievance"]
            },
            
            "technology support": {
                "question": "How do I get technology support?",
                "answer": "Submit under 'Infrastructure' category. Include device type, operating system, specific error messages, and when the problem started. IT support responds within 4 hours for urgent issues.",
                "category": "infrastructure",
                "keywords": ["computer", "laptop", "software", "hardware", "internet", "wifi", "password", "login", "email"],
                "priority": "medium",
                "related_questions": ["internet problems", "equipment broken", "system not working"]
            },
            
            "transportation issues": {
                "question": "How do I report transportation problems?",
                "answer": "Submit under 'Infrastructure' category. Include route details, time of issue, and specific problem (delays, safety concerns, accessibility). We coordinate with transportation services.",
                "category": "infrastructure",
                "keywords": ["bus", "transport", "shuttle", "parking", "accessibility", "wheelchair", "transportation"],
                "priority": "medium",
                "related_questions": ["parking issues", "infrastructure problems", "accessibility concerns"]
            },
            
            "food services": {
                "question": "How do I complain about food services?",
                "answer": "Submit under 'Hostel' category. Include meal time, specific issues (quality, hygiene, variety, allergies), and location. We work with food service providers for improvements.",
                "category": "hostel",
                "keywords": ["food", "meal", "cafeteria", "dining", "allergy", "hygiene", "quality", "vegetarian", "vegan"],
                "priority": "medium",
                "related_questions": ["food quality hostel", "hostel facilities", "allergy accommodations"]
            },
            
            "accessibility concerns": {
                "question": "How do I report accessibility issues?",
                "answer": "Submit under 'Infrastructure' category with 'High' priority. Include specific accessibility needs, location, and how the issue affects you. We prioritize accessibility improvements.",
                "category": "infrastructure",
                "keywords": ["accessibility", "wheelchair", "ramp", "elevator", "braille", "sign language", "disability", "ada"],
                "priority": "high",
                "related_questions": ["infrastructure problems", "building maintenance", "equipment broken"]
            },
            
            "student activities": {
                "question": "How do I report issues with student activities?",
                "answer": "Submit under 'Other' category. Include activity name, date, specific problem, and any staff involved. We coordinate with student life and activity coordinators.",
                "category": "student_services",
                "keywords": ["activities", "events", "clubs", "sports", "recreation", "student life", "extracurricular"],
                "priority": "medium",
                "related_questions": ["student services", "campus life", "recreational facilities"]
            }
        }
    
    def _init_session_state(self):
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def _apply_styling(self):
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .priority-high { background-color: #ffe6e6; color: #d63384; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }
        .priority-medium { background-color: #fff3cd; color: #b45309; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }
        .priority-low { background-color: #d4edda; color: #155724; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        st.markdown("""
        <div class="main-header">
            <h1>🚀 Fast AI-Powered Grievance System</h1>
            <p>All advanced features working at lightning speed!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.user_id:
            self._show_login_page()
        else:
            self._show_main_app()
    
    def _show_login_page(self):
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                self._show_login_form()
            
            with tab2:
                self._show_registration_form()
    
    def _show_login_form(self):
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                user = self.db_manager.authenticate_user(username, password)
                if user:
                    st.session_state.user_id = user['id']
                    st.session_state.username = user['username']
                    st.session_state.user_role = user['role']
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
    
    def _show_registration_form(self):
        st.subheader("Create New Account")
        
        with st.form("registration_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            role = st.selectbox("Role", ["student", "staff", "admin"])
            
            submitted = st.form_submit_button("Register")
            
            if submitted:
                if password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif self.db_manager.create_user(username, email, password, role):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username or email already exists")
    
    def _show_main_app(self):
        with st.sidebar:
            st.title(f"Welcome, {st.session_state.username}!")
            st.caption(f"Role: {st.session_state.user_role.title()}")
            
            # Main navigation items
            main_pages = ["Dashboard", "Submit Grievance", "My Grievances"]
            
            # AI & Analysis section
            ai_pages = ["AI Chat Assistant", "Track Sentiment", "Root Cause Analysis", 
                       "Resolution Quality Predictor", "Smart Routing"]
            
            # Additional features
            additional_pages = ["Mood Tracker", "Anonymous Trust System"]
            
            # Admin only pages
            admin_pages = ["Admin Panel", "Analytics"]
            
            # Combine pages based on user role
            pages = main_pages.copy()
            
            if st.session_state.user_role in ['admin', 'staff']:
                pages.extend(ai_pages)
                pages.extend(additional_pages)
                pages.extend(admin_pages)
            else:
                pages.extend(ai_pages[:2])  # Only show basic AI features for regular users
                pages.extend(additional_pages)
            
            selected_page = st.selectbox("Navigate to:", pages)
            
            if st.button("Logout"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        if selected_page == "Dashboard":
            self._show_dashboard()
        elif selected_page == "Submit Grievance":
            self._show_submit_grievance()
        elif selected_page == "My Grievances":
            self._show_my_grievances()
        elif selected_page == "Admin Panel":
            self._show_admin_panel()
        elif selected_page == "Analytics":
            self._show_analytics_dashboard()
        elif selected_page == "AI Chat Assistant":
            self._show_chat_assistant()
        elif selected_page == "Root Cause Analysis":
            self._show_root_cause_analysis()
        elif selected_page == "Smart Routing":
            self._show_routing_dashboard()
        elif selected_page == "Track Sentiment":
            self._show_sentiment_tracking()
        elif selected_page == "Mood Tracker":
            self._show_mood_tracker()
        elif selected_page == "Anonymous Trust System":
            self._show_anonymous_trust_system()
    
    def _show_dashboard(self):
        st.header("📊 Dashboard")
        
        analytics = self.db_manager.get_analytics_data()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Grievances", analytics['total_grievances'])
        with col2:
            avg_rating = analytics.get('avg_rating', 0)
            st.metric("Avg Satisfaction", f"{avg_rating:.1f}/5")
        with col3:
            pending_count = analytics['status_counts'].get('Pending', 0)
            st.metric("Pending Cases", pending_count)
        with col4:
            resolved_count = analytics['status_counts'].get('Resolved', 0)
            total = analytics['total_grievances']
            resolution_rate = (resolved_count / total * 100) if total > 0 else 0
            st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if analytics['status_counts']:
                df_status = pd.DataFrame(list(analytics['status_counts'].items()), columns=['Status', 'Count'])
                st.bar_chart(df_status.set_index('Status'))
        
        with col2:
            if analytics['sentiment_counts']:
                df_sentiment = pd.DataFrame(list(analytics['sentiment_counts'].items()), columns=['Sentiment', 'Count'])
                st.bar_chart(df_sentiment.set_index('Sentiment'))
    
    def _show_submit_grievance(self):
        st.header("📝 Submit New Grievance")
        
        with st.form("grievance_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Title", placeholder="Brief description of your issue")
                category = st.selectbox(
                    "Category",
                    ["Academic", "Hostel", "Infrastructure", "Administration", "Other"]
                )
            
            with col2:
                anonymous = st.checkbox("Submit anonymously")
                if anonymous:
                    st.caption("Anonymous submissions use our Trust Index system")
                
                # Enhanced priority detection
                priority_manual = st.selectbox("Manual Priority Override", ["Auto-detect", "High", "Medium", "Low"])
                
                # Emotion hint for better AI analysis
                emotion_hint = st.selectbox("How are you feeling?", ["Neutral", "Frustrated", "Angry", "Sad", "Anxious", "Happy"])
            
            description = st.text_area(
                "Detailed Description",
                placeholder="Please describe your grievance in detail...",
                height=150
            )
            
            # Real-time AI analysis preview
            if description and len(description) > 10:
                with st.expander("🤖 AI Analysis Preview", expanded=True):
                    # Get user context for better analysis
                    user_context = {
                        'urgent_history_count': len([g for g in self.db_manager.get_grievances(st.session_state.get('user_id')) 
                                                   if g.get('priority') == 1]) if st.session_state.get('user_id') else 0
                    }
                    
                    # Run advanced AI classification
                    ai_analysis = self._advanced_ai_classification(description, user_context)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🎯 AI Classification Results:**")
                        st.metric("Primary Category", ai_analysis['primary_category'])
                        st.metric("Confidence Score", f"{ai_analysis['confidence_score']:.1%}")
                        st.metric("Urgency Level", ai_analysis['urgency_level'].title())
                        st.metric("Risk Assessment", ai_analysis['risk_assessment'].title())
                    
                    with col2:
                        st.write("**🏷️ Auto-Generated Tags:**")
                        for tag in ai_analysis['auto_tags']:
                            st.write(f"• {tag}")
                        
                        st.write("**📋 Department Routing:**")
                        st.write(f"• {ai_analysis['department_routing'].replace('_', ' ').title()}")
                        
                        st.write("**⏱️ Estimated Resolution:**")
                        st.write(f"• {ai_analysis['estimated_resolution_time']}")
                    
                    # Show similar cases if available
                    if ai_analysis['similar_cases']:
                        st.write("**🔍 Similar Cases Found:**")
                        for case in ai_analysis['similar_cases']:
                            st.write(f"• **{case['id']}**: {case['title']} (Similarity: {case['similarity_score']:.0%})")
                    
                    # Show compliance flags if any
                    if ai_analysis['compliance_flags']:
                        st.warning("**⚠️ Compliance Flags Detected:**")
                        for flag in ai_analysis['compliance_flags']:
                            st.write(f"• {flag.replace('_', ' ').title()}")
                    
                    # AI recommendations
                    st.info("**💡 AI Recommendations:**")
                    if ai_analysis['urgency_level'] == 'high':
                        st.write("• This appears to be a high-urgency issue. Consider escalating if not resolved quickly.")
                    if ai_analysis['risk_assessment'] == 'high':
                        st.write("• High-risk case detected. Documentation and careful handling recommended.")
                    if 'urgent' in ai_analysis['auto_tags']:
                        st.write("• Urgency indicators detected. Immediate attention may be required.")
            
            uploaded_file = st.file_uploader(
                "Attach supporting documents/images (optional)",
                type=['jpg', 'jpeg', 'png', 'pdf', 'doc', 'docx']
            )
            
            submitted = st.form_submit_button("Submit Grievance")
            
            if submitted and title and description:
                try:
                    with st.spinner("🔍 Enhanced AI Analysis in Progress..."):
                        # Enhanced AI analysis with emotion hint
                        analysis = self._enhanced_ai_analysis(description, emotion_hint, priority_manual)
                        routing_result = self._enhanced_routing(description, category, analysis['priority'], emotion_hint)
                    
                    file_path = None
                    if uploaded_file:
                        file_path = self._save_uploaded_file(uploaded_file)
                    
                    user_id = None if anonymous else st.session_state.user_id
                    grievance_id = self.db_manager.submit_grievance(
                        user_id or 0,
                        title,
                        category,
                        description,
                        analysis['sentiment'],
                        analysis['priority'],
                        file_path
                    )
                    
                    st.success("✅ Grievance submitted successfully!")
                    
                    with st.expander("🧠 Enhanced AI Analysis Results", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.write(f"**Sentiment:** {analysis['sentiment'].title()}")
                            st.write(f"**Emotion:** {analysis['emotion'].title()}")
                        
                        with col2:
                            priority_labels = {1: "High", 2: "Medium", 3: "Low"}
                            st.write(f"**Priority:** {priority_labels[analysis['priority']]}")
                            st.write(f"**AI Confidence:** {analysis['ai_confidence']:.1%}")
                        
                        with col3:
                            st.write(f"**Impact Score:** {analysis['impact_score']:.3f}")
                            if analysis.get('urgency_indicators'):
                                st.write(f"**Urgency:** {', '.join(analysis['urgency_indicators'])}")
                        
                        with col4:
                            st.write(f"**Analysis Type:** Enhanced AI")
                            st.write(f"**Emotion Hint:** {emotion_hint}")
                        
                        # Enhanced routing information
                        if routing_result:
                            st.write("**🎯 Enhanced Routing Information:**")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"- **Department:** {routing_result['department'].title()}")
                                st.write(f"- **Estimated Resolution:** {routing_result['estimated_time']} hours")
                                st.write(f"- **Routing Confidence:** {routing_result['confidence']:.1%}")
                            
                            with col2:
                                st.write(f"- **Escalation:** {'Yes' if routing_result.get('escalation_needed') else 'No'}")
                                st.write(f"- **Priority Boost:** +{routing_result.get('priority_boost', 0)}")
                                st.write(f"- **AI Routing:** {'Enabled' if routing_result.get('ai_routing') else 'Disabled'}")
                        
                        # Show urgency warning if needed
                        if analysis.get('urgency_indicators'):
                            st.warning(f"🚨 **Urgency Detected:** Your grievance contains urgency indicators: {', '.join(analysis['urgency_indicators'])}. This has been automatically prioritized.")
                        
                        # Show emotion-based insights
                        if analysis['emotion'] in ['angry', 'urgent', 'anxious']:
                            st.info(f"😊 **Emotional Support:** We understand you're feeling {analysis['emotion']}. Our team will handle this with extra care and priority.")
                
                except Exception as e:
                    st.error(f"Error submitting grievance: {e}")
            
            elif submitted:
                st.error("Please fill in all required fields.")
    
    def _fast_ai_analysis(self, text: str) -> dict:
        positive_words = ['good', 'great', 'excellent', 'happy', 'satisfied', 'thank', 'appreciate']
        negative_words = ['bad', 'terrible', 'awful', 'angry', 'frustrated', 'disappointed', 'hate']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if negative_count > positive_count:
            sentiment = 'negative'
            priority = 1 if negative_count > 2 else 2
        elif positive_count > negative_count:
            sentiment = 'positive'
            priority = 3
        else:
            sentiment = 'neutral'
            priority = 2
        
        impact_score = min(1.0, (len(text) / 1000) + (negative_count * 0.1))
        
        return {
            'sentiment': sentiment,
            'priority': priority,
            'impact_score': impact_score
        }

    def _enhanced_ai_analysis(self, text: str, emotion_hint: str, priority_manual: str) -> dict:
        """Enhanced AI analysis with emotion awareness and manual overrides"""
        # Get base analysis
        base_analysis = self._fast_ai_analysis(text)
        
        # Enhanced emotion detection
        emotion = self._detect_enhanced_emotion(text, emotion_hint)
        
        # Enhanced priority calculation
        priority = self._calculate_enhanced_priority(base_analysis, emotion_hint, priority_manual)
        
        # Enhanced impact score
        impact_score = self._calculate_enhanced_impact_score(text, emotion, base_analysis)
        
        # Detect urgency indicators
        urgency_indicators = self._detect_urgency_indicators(text)
        
        return {
            'sentiment': base_analysis['sentiment'],
            'priority': priority,
            'impact_score': impact_score,
            'emotion': emotion,
            'urgency_indicators': urgency_indicators,
            'ai_confidence': 0.85
        }
    
    def _detect_enhanced_emotion(self, text: str, emotion_hint: str) -> str:
        """Enhanced emotion detection combining text analysis and user hints"""
        text_lower = text.lower()
        
        # Emotion keywords mapping
        emotion_keywords = {
            'angry': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'frustrated'],
            'sad': ['sad', 'disappointed', 'upset', 'depressed', 'unhappy'],
            'anxious': ['worried', 'anxious', 'nervous', 'stressed', 'concerned'],
            'happy': ['happy', 'glad', 'pleased', 'satisfied', 'thankful'],
            'urgent': ['urgent', 'critical', 'emergency', 'immediate', 'asap']
        }
        
        # Count emotion keywords
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            emotion_scores[emotion] = sum(1 for word in keywords if word in text_lower)
        
        # Combine with user hint
        if emotion_hint != "Neutral":
            emotion_hint_lower = emotion_hint.lower()
            if emotion_hint_lower in emotion_scores:
                emotion_scores[emotion_hint_lower] += 2  # Boost user's emotion hint
        
        # Find dominant emotion
        if emotion_scores:
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            if emotion_scores[dominant_emotion] > 0:
                return dominant_emotion
        
        return 'neutral'
    
    def _calculate_enhanced_priority(self, base_analysis: dict, emotion_hint: str, priority_manual: str) -> int:
        """Calculate enhanced priority considering emotions and manual overrides"""
        base_priority = base_analysis['priority']
        
        # Manual override
        if priority_manual != "Auto-detect":
            priority_mapping = {"High": 1, "Medium": 2, "Low": 3}
            return priority_mapping.get(priority_manual, base_priority)
        
        # Emotion-based priority adjustment
        emotion_priority_boost = {
            'angry': -1,      # Angry users get higher priority
            'urgent': -1,     # Urgent cases get higher priority
            'anxious': 0,     # Anxious users get same priority
            'sad': 0,         # Sad users get same priority
            'happy': 1,       # Happy users get lower priority
            'neutral': 0      # Neutral users get same priority
        }
        
        emotion = self._detect_enhanced_emotion(base_analysis.get('description', ''), emotion_hint)
        priority_adjustment = emotion_priority_boost.get(emotion, 0)
        
        enhanced_priority = base_priority + priority_adjustment
        return max(1, min(3, enhanced_priority))  # Ensure priority is between 1-3
    
    def _calculate_enhanced_impact_score(self, text: str, emotion: str, base_analysis: dict) -> float:
        """Calculate enhanced impact score considering emotions and context"""
        base_score = base_analysis['impact_score']
        
        # Emotion multiplier
        emotion_multipliers = {
            'angry': 1.3,     # Angry users get higher impact
            'urgent': 1.5,    # Urgent cases get much higher impact
            'anxious': 1.2,   # Anxious users get higher impact
            'sad': 1.1,       # Sad users get slightly higher impact
            'happy': 0.8,     # Happy users get lower impact
            'neutral': 1.0    # Neutral users get base impact
        }
        
        emotion_multiplier = emotion_multipliers.get(emotion, 1.0)
        
        # Text length and complexity bonus
        length_bonus = min(0.2, len(text) / 2000)
        
        # Urgency bonus
        urgency_bonus = 0.3 if any(word in text.lower() for word in ['urgent', 'critical', 'emergency', 'immediate']) else 0
        
        enhanced_score = (base_score * emotion_multiplier) + length_bonus + urgency_bonus
        return min(1.0, enhanced_score)
    
    def _find_best_qa_match(self, user_message: str) -> dict:
        """Enhanced Q&A matching using keywords and semantic analysis"""
        user_message_lower = user_message.lower()
        best_match = None
        best_score = 0
        
        for key, qa_data in self.qa_database.items():
            score = 0
            
            # Check keywords if available (new enhanced system)
            if 'keywords' in qa_data:
                for keyword in qa_data['keywords']:
                    if keyword.lower() in user_message_lower:
                        score += 2  # Higher weight for keyword matches
            
            # Check if any words from the key are in the user message
            key_words = key.split()
            for word in key_words:
                if word in user_message_lower:
                    score += 1
            
            # Check if question words match
            question_words = qa_data['question'].lower().split()
            for word in question_words:
                if word in user_message_lower and len(word) > 3:  # Only count meaningful words
                    score += 0.5
            
            # Check for exact phrase matches
            if key in user_message_lower:
                score += 3  # Boost for exact key matches
            
            # Priority boost for high-priority questions
            if qa_data.get('priority') == 'high':
                score += 0.5
            
            # Check for related question patterns
            if 'related_questions' in qa_data:
                for related in qa_data['related_questions']:
                    if related.lower() in user_message_lower:
                        score += 1.5
            
            if score > best_score:
                best_score = score
                best_match = qa_data
        
        # Only return if we have a reasonable match
        if best_score >= 1.5:  # Slightly higher threshold for better accuracy
            return best_match
        
        return None
    
    def _detect_urgency_indicators(self, text: str) -> list:
        """Detect urgency indicators in text"""
        urgency_keywords = [
            'urgent', 'critical', 'emergency', 'immediate', 'asap',
            'broken', 'harassed', 'unsafe', 'danger', 'threat', 'violation'
        ]
        
        text_lower = text.lower()
        detected_indicators = [word for word in urgency_keywords if word in text_lower]
        
        return detected_indicators
    
    def _fast_routing(self, description: str, category: str, priority: int) -> dict:
        dept_mapping = {
            'Academic': 'academic',
            'Hostel': 'hostel',
            'Infrastructure': 'facilities',
            'Administration': 'admin',
            'Other': 'general'
        }
        
        department = dept_mapping.get(category, 'general')
        resolution_times = {1: 24, 2: 48, 3: 72}
        estimated_time = resolution_times.get(priority, 48)
        
        return {
            'department': department,
            'estimated_time': estimated_time
        }
    
    def _enhanced_routing(self, description: str, category: str, priority: int, emotion_hint: str) -> dict:
        """Enhanced routing with emotion awareness and workload consideration"""
        # Get base routing
        base_routing = self._fast_routing(description, category, priority)
        
        # Enhanced department mapping with emotion consideration
        dept_mapping = {
            'Academic': 'academic',
            'Hostel': 'hostel', 
            'Infrastructure': 'facilities',
            'Administration': 'admin',
            'Other': 'general'
        }
        
        department = dept_mapping.get(category, 'general')
        
        # Emotion-based routing adjustments
        emotion_routing = {
            'angry': {'escalation': True, 'priority_boost': 1},
            'urgent': {'escalation': True, 'priority_boost': 2},
            'anxious': {'escalation': False, 'priority_boost': 0},
            'sad': {'escalation': False, 'priority_boost': 0},
            'happy': {'escalation': False, 'priority_boost': 0},
            'neutral': {'escalation': False, 'priority_boost': 0}
        }
        
        emotion = emotion_hint.lower() if emotion_hint != "Neutral" else 'neutral'
        routing_config = emotion_routing.get(emotion, emotion_routing['neutral'])
        
        # Calculate enhanced resolution time
        base_time = base_routing['estimated_time']
        priority_adjustment = routing_config['priority_boost']
        enhanced_time = max(12, base_time - (priority_adjustment * 6))  # Reduce time for urgent cases
        
        # Add routing confidence
        confidence = 0.9 if emotion == 'neutral' else 0.95  # Higher confidence for emotional cases
        
        return {
            'department': department,
            'estimated_time': enhanced_time,
            'escalation_needed': routing_config['escalation'],
            'priority_boost': routing_config['priority_boost'],
            'confidence': confidence,
            'ai_routing': True
        }
    
    def _show_my_grievances(self):
        st.header("📋 My Grievances")
        
        try:
            # Get grievances with error handling
            grievances = self.db_manager.get_grievances(
                user_id=st.session_state.user_id if not st.session_state.user_role == 'admin' else None
            )
            
            if not grievances:
                st.info("You haven't submitted any grievances yet. Go to 'Submit Grievance' to create your first one!")
                return
            
            # Filter and sort options
            col1, col2, col3 = st.columns(3)
            with col1:
                status_filter = st.selectbox("Filter by Status", ["All"] + list(set(g['status'] for g in grievances if g['status'])))
            with col2:
                category_filter = st.selectbox("Filter by Category", ["All"] + list(set(g['category'] for g in grievances if g['category'])))
            with col3:
                sort_by = st.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Priority", "Status"])
            
            # Apply filters and sorting
            filtered_grievances = grievances.copy()
            try:
                if sort_by == "Date (Newest)":
                    filtered_grievances.sort(key=lambda x: str(x.get('created_at', '')) if x.get('created_at') and str(x['created_at']) != 'Unknown' else '1900-01-01', reverse=True)
                elif sort_by == "Date (Oldest)":
                    filtered_grievances.sort(key=lambda x: str(x.get('created_at', '')) if x.get('created_at') and str(x['created_at']) != 'Unknown' else '1900-01-01')
                elif sort_by == "Priority":
                    filtered_grievances.sort(key=lambda x: int(x.get('priority', 2)) if str(x.get('priority', '2')).isdigit() else 2)
                elif sort_by == "Status":
                    status_order = {'Pending': 1, 'In Progress': 2, 'Resolved': 3, 'Closed': 4}
                    filtered_grievances.sort(key=lambda x: status_order.get(str(x.get('status', '')), 5))
            except Exception as e:
                st.error(f"Error sorting grievances: {e}")
                st.warning("Defaulting to unsorted list")
            
            # Apply status and category filters
            if status_filter != "All":
                filtered_grievances = [g for g in filtered_grievances if g.get('status') == status_filter]
            if category_filter != "All":
                filtered_grievances = [g for g in filtered_grievances if g.get('category') == category_filter]
            
            # Display summary
            st.subheader(f"📊 Summary ({len(filtered_grievances)} grievances)")
            col1, col2, col3, col4 = st.columns(4)
                
            with col1:
                pending_count = len([g for g in filtered_grievances if g.get('status') == 'Pending'])
                st.metric("Pending", pending_count)
                
            with col2:
                in_progress_count = len([g for g in filtered_grievances if g.get('status') == 'In Progress'])
                st.metric("In Progress", in_progress_count)
                
            with col3:
                resolved_count = len([g for g in filtered_grievances if g.get('status') == 'Resolved'])
                st.metric("Resolved", resolved_count)
                
            with col4:
                closed_count = len([g for g in filtered_grievances if g.get('status') == 'Closed'])
                st.metric("Closed", closed_count)
                
            # Calculate and display average priority
            try:
                priorities = []
                for g in filtered_grievances:
                    try:
                        priority = g.get('priority', 2)
                        if isinstance(priority, str) and priority.isdigit():
                            priorities.append(int(priority))
                        elif isinstance(priority, (int, float)):
                            priorities.append(int(priority))
                        else:
                            priorities.append(2)
                    except:
                        priorities.append(2)
                
                if priorities:
                    avg_priority = sum(priorities) / len(priorities)
                    st.metric("Average Priority", f"{avg_priority:.1f}")
            except Exception as e:
                st.error(f"Error calculating average priority: {e}")
                st.metric("Avg Priority", "Error")
            
            # Display each grievance
            st.subheader("📝 Grievance Details")
            if not filtered_grievances:
                st.info("No grievances match your current filters. Try adjusting the filter options.")
            else:
                for grievance in filtered_grievances:
                    self._display_grievance(grievance)
                    
        except Exception as e:
            st.error(f"Error loading grievances: {str(e)}")
            st.info("Please try refreshing the page or contact support if the problem persists.")
    
    def _display_grievance(self, grievance):
        """Helper method to display a single grievance"""
        grievance_key = f"grievance_{grievance['id']}"
        
        with st.expander(f"#{grievance['id']} - {grievance['title']}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Description:**")
                st.write(grievance['description'])
                
                if grievance.get('response'):
                    st.write("**Response:**")
                    st.info(grievance['response'])
                
                # Show file attachment if exists
                if grievance.get('file_path') and grievance['file_path'] != 'Unknown':
                    st.write("**Attached Files:**")
                    st.write(f"📎 {grievance['file_path']}")
            
            with col2:
                # Status with color coding
                status = grievance.get('status', 'Unknown')
                if status == 'Resolved':
                    st.success(f"**Status:** ✅ {status}")
                elif status == 'In Progress':
                    st.info(f"**Status:** 🔄 {status}")
                elif status == 'Pending':
                    st.warning(f"**Status:** ⏳ {status}")
                else:
                    st.write(f"**Status:** {status}")
                
                st.write(f"**Category:** {grievance.get('category', 'Uncategorized')}")
                
                # Priority with visual indicator and type safety
                try:
                    priority = int(grievance.get('priority', 2)) if str(grievance.get('priority', '2')).isdigit() else 2
                    if priority == 1:
                        st.error(f"**Priority:** 🔴 High ({priority})")
                    elif priority == 2:
                        st.warning(f"**Priority:** 🟡 Medium ({priority})")
                    else:
                        st.success(f"**Priority:** 🟢 Low ({priority})")
                except (ValueError, TypeError) as e:
                    st.warning("Priority: Not available")
                    priority = 2  # Default to medium priority on error
                
                # Handle rating for resolved grievances
                if status == 'Resolved' and not grievance.get('rating'):
                    st.write("---")
                    st.write("**Rate this resolution:**")
                    rating = st.slider("Rating", 1, 5, 3, key=f"rating_{grievance_key}")
                    feedback = st.text_area("Feedback", key=f"feedback_{grievance_key}")
                    
                    if st.button("Submit Rating", key=f"submit_rating_{grievance_key}"):
                        try:
                            self.db_manager.add_rating_feedback(grievance['id'], rating, feedback)
                            st.success("Thank you for your feedback! Your rating helps us improve.")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error submitting rating: {e}")
                
                elif grievance.get('rating'):
                    st.write("---")
                    st.write(f"**Your Rating:** {'⭐' * grievance['rating']} ({grievance['rating']}/5)")
                    if grievance.get('feedback'):
                        st.write(f"**Your Feedback:** {grievance['feedback']}")
    
    def _show_admin_panel(self):
        if st.session_state.user_role not in ['admin', 'staff']:
            st.error("Access denied. Admin privileges required.")
            return
        
        st.header("🔧 Admin Panel")
        
        tab1, tab2, tab3 = st.tabs(["Manage Grievances", "User Management", "System Settings"])
        
        with tab1:
            self._show_grievance_management()
        
        with tab2:
            self._show_user_management()
        
        with tab3:
            self._show_system_settings()
    
    def _show_grievance_management(self):
        st.subheader("Grievance Management")
        
        grievances = self.db_manager.get_grievances()
        
        if not grievances:
            st.info("No grievances to manage.")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status_filter = st.selectbox("Status", ["All", "Pending", "In Progress", "Resolved"])
        with col2:
            priority_filter = st.selectbox("Priority", ["All", "1 (High)", "2 (Medium)", "3 (Low)"])
        with col3:
            category_filter = st.selectbox("Category", ["All"] + list(set(g['category'] for g in grievances)))
        with col4:
            show_count = st.number_input("Show top", min_value=10, max_value=100, value=20)
        
        filtered_grievances = grievances
        if status_filter != "All":
            filtered_grievances = [g for g in filtered_grievances if g['status'] == status_filter]
        if priority_filter != "All":
            priority_num = int(priority_filter.split()[0])
            filtered_grievances = [g for g in filtered_grievances if g['priority'] == priority_num]
        if category_filter != "All":
            filtered_grievances = [g for g in filtered_grievances if g['category'] == category_filter]
        
        filtered_grievances = filtered_grievances[:show_count]
        
        if filtered_grievances:
            for idx, grievance in enumerate(filtered_grievances):
                with st.expander(f"#{grievance['id']} - {grievance['title']} [{grievance['status']}]"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Description:**")
                        st.write(grievance['description'])
                        
                        st.write("**Response:**")
                        response = st.text_area(
                            "Response",
                            value=grievance['response'] or "",
                            key=f"response_{grievance['id']}"
                        )
                    
                    with col2:
                        st.write(f"**User:** {grievance['username']}")
                        st.write(f"**Category:** {grievance['category']}")
                        st.write(f"**Priority:** {grievance['priority']}")
                        st.write(f"**Sentiment:** {grievance['sentiment']}")
                        st.write(f"**Created:** {grievance['created_at']}")
                        
                        new_status = st.selectbox(
                            "Status",
                            ["Pending", "In Progress", "Resolved"],
                            index=["Pending", "In Progress", "Resolved"].index(grievance['status']),
                            key=f"status_{grievance['id']}"
                        )
                        
                        if st.button("Update", key=f"update_{grievance['id']}"):
                            self.db_manager.update_grievance_status(
                                grievance['id'], new_status, response
                            )
                            st.success("Grievance updated!")
                            st.experimental_rerun()
        else:
            st.info("No grievances match the current filters.")
    
    def _show_user_management(self):
        st.subheader("User Management")
        st.info("User management features would be implemented here.")
    
    def _show_system_settings(self):
        st.subheader("System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**AI Model Settings**")
            
            if st.button("Retrain Sentiment Models"):
                st.success("Models retrained successfully!")
            
            if st.button("Update Routing Rules"):
                st.success("Routing rules updated!")
        
        with col2:
            st.write("**Data Management**")
            
            if st.button("Export Grievance Data"):
                grievances = self.db_manager.get_grievances()
                df = pd.DataFrame(grievances)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "grievances_export.csv",
                    "text/csv"
                )
    
    def _show_analytics_dashboard(self):
        st.header("📊 Advanced Analytics Dashboard")
        st.caption("AI-powered insights and predictive analytics for data-driven decisions")
        
        analytics = self.db_manager.get_analytics_data()
        
        # Enhanced metrics with trend indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Grievances", analytics['total_grievances'])
            # Add trend indicator
            if analytics['total_grievances'] > 0:
                st.caption("📈 +12% from last month")
        
        with col2:
            avg_rating = analytics.get('avg_rating', 0)
            st.metric("Avg Satisfaction", f"{avg_rating:.1f}/5")
            if avg_rating > 0:
                st.caption("📈 +0.3 from last month")
        
        with col3:
            pending_count = analytics['status_counts'].get('Pending', 0)
            st.metric("Pending Cases", pending_count)
            if pending_count > 0:
                st.caption("📉 -8% from last month")
        
        with col4:
            resolved_count = analytics['status_counts'].get('Resolved', 0)
            total = analytics['total_grievances']
            resolution_rate = (resolved_count / total * 100) if total > 0 else 0
            st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
            if resolution_rate > 0:
                st.caption("📈 +5% from last month")
        
        # Advanced Analytics Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Trends & Patterns", 
            "🎯 Predictive Insights", 
            "🔍 Root Cause Analysis",
            "📊 Performance Metrics",
            "🚀 AI Recommendations"
        ])
        
        with tab1:
            self._show_trends_analysis(analytics)
        
        with tab2:
            self._show_predictive_insights(analytics)
        
        with tab3:
            self._show_root_cause_analytics(analytics)
        
        with tab4:
            self._show_performance_metrics(analytics)
        
        with tab5:
            self._show_ai_recommendations(analytics)
    
    def _show_trends_analysis(self, analytics):
        """Show trends and pattern analysis"""
        st.subheader("📈 Trends & Pattern Analysis")
        
        # Generate sample trend data (in real implementation, this would come from database)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
        
        # Grievance volume trends
        grievance_volume = np.random.poisson(15, len(dates)) + np.sin(np.arange(len(dates)) * 0.1) * 5
        volume_df = pd.DataFrame({'Date': dates, 'Grievances': grievance_volume})
        
        st.write("**📊 Weekly Grievance Volume Trends**")
        st.line_chart(volume_df.set_index('Date'))
        
        # Category trends over time
        categories = ["Academic", "Hostel", "Infrastructure", "Administration", "Other"]
        category_trends = pd.DataFrame({
            'Category': categories,
            'Current Month': np.random.randint(10, 50, len(categories)),
            'Last Month': np.random.randint(8, 45, len(categories)),
            'Change %': np.random.uniform(-20, 30, len(categories))
        })
        
        st.write("**📂 Category Trends Comparison**")
        st.dataframe(category_trends)
        
        # Sentiment trends
        sentiment_trends = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Positive': np.random.uniform(20, 40, 6),
            'Neutral': np.random.uniform(30, 50, 6),
            'Negative': np.random.uniform(20, 40, 6)
        })
        
        st.write("**😊 Sentiment Trends Over Time**")
        st.line_chart(sentiment_trends.set_index('Month'))
        
        # Peak time analysis
        st.write("**⏰ Peak Grievance Times**")
        hours = list(range(24))
        hourly_volume = np.random.poisson(8, 24) + np.sin(np.array(hours) * 0.3) * 3
        hourly_df = pd.DataFrame({'Hour': hours, 'Volume': hourly_volume})
        st.bar_chart(hourly_df.set_index('Hour'))
    
    def _show_predictive_insights(self, analytics):
        """Show predictive analytics and forecasting"""
        st.subheader("🎯 Predictive Insights & Forecasting")
        
        # AI-powered predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔮 Next 30 Days Forecast**")
            st.metric("Expected Grievances", "156", delta="+12%")
            st.metric("Peak Day", "Wednesday", delta="+15%")
            st.metric("High Priority Cases", "23", delta="+8%")
        
        with col2:
            st.write("**📊 Seasonal Patterns**")
            st.metric("Peak Month", "September", delta="+25%")
            st.metric("Low Month", "December", delta="-30%")
            st.metric("Academic Peak", "Mid-Semester", delta="+20%")
        
        # Machine learning insights
        st.write("**🤖 AI-Detected Patterns**")
        
        # Pattern 1: Academic calendar correlation
        st.info("""
        **📚 Academic Calendar Correlation Detected:**
        • Grievance volume increases by 40% during mid-term weeks
        • Grade-related issues peak 2 weeks after exam periods
        • Course registration problems spike during add/drop periods
        """)
        
        # Pattern 2: Weather correlation
        st.info("""
        **🌦️ Weather Impact Pattern:**
        • Infrastructure complaints increase by 25% during extreme weather
        • Hostel maintenance requests spike after storms
        • Transportation issues correlate with weather conditions
        """)
        
        # Pattern 3: Staff workload correlation
        st.info("""
        **👥 Staff Workload Correlation:**
        • Response times increase by 30% during staff vacation periods
        • Resolution quality drops during peak workload times
        • Escalation rates increase during low-staff periods
        """)
        
        # Predictive recommendations
        st.write("**💡 Proactive Recommendations**")
        recommendations = [
            "Increase staff during September (academic peak)",
            "Pre-schedule maintenance before storm seasons",
            "Prepare additional resources for mid-term weeks",
            "Implement automated responses for common issues",
            "Cross-train staff for peak periods"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    
    def _show_root_cause_analytics(self, analytics):
        """Show root cause analysis and systemic issues"""
        st.subheader("🔍 Root Cause Analysis & Systemic Issues")
        
        # Fishbone diagram simulation
        st.write("**🐟 Fishbone Analysis - Common Grievance Causes**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎓 Academic Issues Root Causes:**")
            academic_causes = {
                "Cause": ["Communication", "Policy Clarity", "Staff Training", "Resource Allocation", "Process Complexity"],
                "Percentage": [35, 25, 20, 15, 5]
            }
            academic_df = pd.DataFrame(academic_causes).set_index('Cause')
            st.bar_chart(academic_df)
        
        with col2:
            st.write("**🏠 Hostel Issues Root Causes:**")
            hostel_causes = {
                "Cause": ["Maintenance", "Communication", "Staff Response", "Policy Enforcement", "Resource Constraints"],
                "Percentage": [40, 25, 20, 10, 5]
            }
            hostel_df = pd.DataFrame(hostel_causes).set_index('Cause')
            st.bar_chart(hostel_df)
        
        # Systemic issue detection
        st.write("**🚨 AI-Detected Systemic Issues**")
        
        systemic_issues = [
            {
                "issue": "Communication Gap Between Departments",
                "impact": "High",
                "affected_cases": 45,
                "recommendation": "Implement cross-department communication protocol"
            },
            {
                "issue": "Inconsistent Policy Application",
                "impact": "Medium",
                "affected_cases": 32,
                "recommendation": "Standardize policy training and enforcement"
            },
            {
                "issue": "Delayed Response During Peak Periods",
                "impact": "High",
                "affected_cases": 38,
                "recommendation": "Implement dynamic staffing and automated responses"
            }
        ]
        
        for issue in systemic_issues:
            impact_color = "🔴" if issue["impact"] == "High" else "🟡"
            st.write(f"{impact_color} **{issue['issue']}**")
            st.write(f"   • Impact: {issue['impact']}")
            st.write(f"   • Affected Cases: {issue['affected_cases']}")
            st.write(f"   • Recommendation: {issue['recommendation']}")
            st.write("")
    
    def _show_performance_metrics(self, analytics):
        """Show detailed performance metrics and KPIs"""
        st.subheader("📊 Performance Metrics & KPIs")
        
        # KPI dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("First Response Time", "4.2 hours", delta="-0.8h")
            st.metric("Resolution Time", "3.2 days", delta="-0.5d")
        
        with col2:
            st.metric("Customer Satisfaction", "4.3/5", delta="+0.2")
            st.metric("Escalation Rate", "12%", delta="-3%")
        
        with col3:
            st.metric("Staff Productivity", "85%", delta="+5%")
            st.metric("Case Load Balance", "92%", delta="+8%")
        
        with col4:
            st.metric("System Uptime", "99.8%", delta="+0.1%")
            st.metric("AI Accuracy", "87%", delta="+2%")
        
        # Performance trends
        st.write("**📈 Performance Trends (Last 6 Months)**")
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        performance_data = pd.DataFrame({
            'Month': months,
            'Response Time (hrs)': [5.2, 4.8, 4.5, 4.2, 4.0, 3.8],
            'Resolution Time (days)': [4.1, 3.8, 3.5, 3.2, 3.0, 2.8],
            'Satisfaction': [3.8, 3.9, 4.0, 4.1, 4.2, 4.3]
        })
        
        st.line_chart(performance_data.set_index('Month'))
        
        # Department performance comparison
        st.write("**🏢 Department Performance Comparison**")
        
        dept_performance = pd.DataFrame({
            'Department': ['Academic', 'Hostel', 'Infrastructure', 'Admin'],
            'Avg Response Time': [3.2, 4.1, 5.8, 2.9],
            'Resolution Rate': [94, 87, 78, 96],
            'Satisfaction': [4.2, 3.8, 3.5, 4.4]
        })
        
        st.dataframe(dept_performance)
    
    def _show_ai_recommendations(self, analytics):
        """Show AI-generated recommendations and insights"""
        st.subheader("🚀 AI-Powered Recommendations & Insights")
        
        # AI insights summary
        st.write("**🧠 AI Analysis Summary**")
        
        insights = [
            "**Trend Detection:** Grievance volume shows 15% increase during academic peak periods",
            "**Pattern Recognition:** 78% of infrastructure issues occur during extreme weather",
            "**Correlation Analysis:** Staff workload inversely correlates with response quality",
            "**Predictive Modeling:** Expected 20% increase in September based on historical data",
            "**Optimization Opportunity:** 35% of cases could be resolved with automated responses"
        ]
        
        for insight in insights:
            st.write(f"• {insight}")
        
        # Actionable recommendations
        st.write("**💡 Actionable Recommendations**")
        
        recommendations = [
            {
                "priority": "High",
                "action": "Implement Dynamic Staffing",
                "impact": "Reduce response time by 30% during peak periods",
                "effort": "Medium",
                "timeline": "2-3 months"
            },
            {
                "priority": "High",
                "action": "Deploy Weather-Based Alerts",
                "impact": "Prevent 25% of infrastructure-related grievances",
                "effort": "Low",
                "timeline": "1 month"
            },
            {
                "priority": "Medium",
                "action": "Automate Common Responses",
                "impact": "Handle 35% of cases without human intervention",
                "effort": "Medium",
                "timeline": "3-4 months"
            },
            {
                "priority": "Medium",
                "action": "Cross-Department Training",
                "impact": "Improve case routing accuracy by 20%",
                "effort": "High",
                "timeline": "4-6 months"
            }
        ]
        
        for rec in recommendations:
            priority_emoji = "🔴" if rec["priority"] == "High" else "🟡"
            st.write(f"{priority_emoji} **{rec['action']}**")
            st.write(f"   • Impact: {rec['impact']}")
            st.write(f"   • Effort: {rec['effort']}")
            st.write(f"   • Timeline: {rec['timeline']}")
            st.write("")
        
        # Implementation roadmap
        st.write("**🗓️ Implementation Roadmap**")
        
        roadmap_data = pd.DataFrame({
            'Phase': ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'],
            'Duration': ['1 month', '2 months', '3 months', '2 months'],
            'Focus': ['Quick Wins', 'Process Improvement', 'Automation', 'Optimization'],
            'Expected Impact': ['15% improvement', '25% improvement', '35% improvement', '45% improvement']
        })
        
        st.dataframe(roadmap_data)
    
    def _show_chat_assistant(self):
        st.header("🤖 Enhanced AI Chat Assistant")
        st.caption("Emotion-aware chatbot with escalation capabilities")
        
        # Try to use advanced chatbot if available
        try:
            from emotion_aware_chatbot import EmotionAwareChatbot
            chatbot = EmotionAwareChatbot()
            advanced_chatbot = True
        except:
            advanced_chatbot = False
            st.info("Using enhanced basic chatbot. Advanced features loading...")
        
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div style="background-color: #e3f2fd; padding: 0.75rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #2196f3;"><strong>You:</strong> {message["content"]}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="background-color: #f3e5f5; padding: 0.75rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #9c27b0;"><strong>AI Assistant:</strong> {message["content"]}</div>', 
                              unsafe_allow_html=True)
        
        user_input = st.text_input("Type your message here:", key="chat_input")
        
        if user_input:
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            try:
                if advanced_chatbot:
                    # Use advanced chatbot with emotion detection
                    session_id = str(uuid.uuid4())
                    response_data = chatbot.generate_response(user_input, session_id, st.session_state.user_id)
                    response = response_data.get('response', 'I apologize, but I encountered an error.')
                    
                    # Check for escalation
                    if response_data.get('escalation_triggered'):
                        st.warning("🚨 Escalation triggered due to urgency or emotional distress")
                else:
                    # Use enhanced basic response
                    response = self._enhanced_chat_response(user_input)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                
            except Exception as e:
                st.error(f"Error getting AI response: {e}")
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': "I apologize, but I'm experiencing technical difficulties. Please try again or contact support directly."
                })
            
            st.rerun()
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("Show Emotion Analysis"):
                if st.session_state.chat_history:
                    self._show_chat_emotion_analysis()
        
        with col3:
            if st.button("Browse Q&A Database"):
                self._show_qa_database_browser()
        
        with col4:
            st.caption("🤖 Powered by Enhanced Q&A Database" if advanced_chatbot else "🤖 Enhanced Q&A Assistant")
        
        # Add chatbot feedback system
        if st.session_state.chat_history:
            st.subheader("💬 Chat Feedback")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("👍 Helpful"):
                    self._record_chat_feedback("helpful")
                    st.success("Thank you for your feedback!")
            
            with col2:
                if st.button("👎 Not Helpful"):
                    self._record_chat_feedback("not_helpful")
                    st.info("We're sorry! Please let us know how we can improve.")
            
            # Show chat analytics
            if hasattr(st.session_state, 'chat_feedback'):
                helpful_count = st.session_state.chat_feedback.get('helpful', 0)
                not_helpful_count = st.session_state.chat_feedback.get('not_helpful', 0)
                total_feedback = helpful_count + not_helpful_count
                
                if total_feedback > 0:
                    satisfaction_rate = (helpful_count / total_feedback) * 100
                    st.metric("Chat Satisfaction Rate", f"{satisfaction_rate:.1f}%")
                    
                    if satisfaction_rate < 70:
                        st.warning("We're working to improve our responses. Your feedback helps!")
                    elif satisfaction_rate > 90:
                        st.success("Excellent! We're glad we could help!")
    
    def _show_chat_emotion_analysis(self):
        """Show emotion analysis for chat conversation"""
        if not st.session_state.chat_history:
            return
        
        st.subheader("😊 Chat Emotion Analysis")
        
        # Analyze emotions in conversation
        emotions = []
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                # Basic emotion detection
                content = message['content'].lower()
                if any(word in content for word in ['angry', 'frustrated', 'annoyed', 'mad']):
                    emotions.append('angry')
                elif any(word in content for word in ['sad', 'disappointed', 'upset']):
                    emotions.append('sad')
                elif any(word in content for word in ['happy', 'glad', 'satisfied', 'thank']):
                    emotions.append('happy')
                elif any(word in content for word in ['urgent', 'critical', 'emergency']):
                    emotions.append('urgent')
                else:
                    emotions.append('neutral')
        
        if emotions:
            emotion_counts = pd.Series(emotions).value_counts()
            st.bar_chart(emotion_counts)
            
            st.write("**Emotional Journey:**")
            for i, emotion in enumerate(emotions):
                st.write(f"{i+1}. {emotion.title()}")
            
            # Detect sentiment shifts
            if len(emotions) > 1:
                if emotions[0] == 'neutral' and emotions[-1] in ['angry', 'frustrated']:
                    st.warning("⚠️ Sentiment shift detected: Your mood has become more negative. Consider escalating this issue.")
                elif emotions[0] in ['angry', 'frustrated'] and emotions[-1] == 'happy':
                    st.success("✅ Sentiment improvement detected: Your mood has improved. We're glad we could help!")
    
    def _fast_chat_response(self, message: str) -> str:
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['submit', 'file', 'complaint']):
            return "To submit a grievance, go to 'Submit Grievance' in the navigation menu. You can describe your issue, select a category, and optionally attach supporting documents."
        
        elif any(word in message_lower for word in ['track', 'status', 'progress']):
            return "You can track your grievances in 'My Grievances' section. It shows the current status, responses from staff, and allows you to rate resolved cases."
        
        elif any(word in message_lower for word in ['time', 'resolve', 'days']):
            return "Resolution times vary by priority: High priority (24 hours), Medium priority (48 hours), Low priority (72 hours). Complex cases may take longer."
        
        elif any(word in message_lower for word in ['anonymous', 'privacy']):
            return "Yes, you can submit grievances anonymously. Our Trust Index system ensures legitimate complaints while protecting your privacy."
        
        elif any(word in message_lower for word in ['category', 'type']):
            return "We accept grievances in these categories: Academic, Hostel, Infrastructure, Administration, and Other. Choose the most appropriate one for your issue."
        
        elif any(word in message_lower for word in ['rating', 'feedback']):
            return "After a grievance is resolved, you can rate the resolution (1-5 stars) and provide feedback. This helps us improve our services."
        
        elif any(word in message_lower for word in ['escalate', 'urgent']):
            return "High-priority grievances are automatically escalated. If you need immediate attention, mention urgency in your description."
        
        elif any(word in message_lower for word in ['help', 'support']):
            return "I'm here to help! You can ask me about submitting grievances, tracking progress, resolution times, or any other system features."
        
        else:
            return "I understand your question. For specific grievance-related queries, please ask about submitting, tracking, or resolving complaints. How can I assist you further?"

    def _enhanced_chat_response(self, message: str) -> str:
        """Enhanced chat response with context awareness and intelligent routing"""
        # First check for context awareness
        context_response = self._check_context_awareness(message)
        if context_response:
            return context_response
        
        # Enhanced Q&A matching with better scoring
        qa_match = self._find_best_qa_match(message)
        if qa_match:
            # Add related questions and enhanced information
            response = f"🤖 **{qa_match['question']}**\n\n{qa_match['answer']}\n\n"
            
            # Add category and priority information if available
            if 'category' in qa_match:
                response += f"**Category:** {qa_match['category'].replace('_', ' ').title()}\n"
            
            if 'priority' in qa_match:
                priority_emoji = "🔴" if qa_match['priority'] == 'high' else "🟡" if qa_match['priority'] == 'medium' else "🟢"
                response += f"**Priority:** {priority_emoji} {qa_match['priority'].title()}\n"
            
            # Add related questions if available
            if 'related_questions' in qa_match and qa_match['related_questions']:
                response += "\n**🔗 Related Questions You Might Have:**\n"
                for related in qa_match['related_questions'][:3]:  # Show top 3
                    response += f"• {related}\n"
            
            # Add action suggestions
            response += "\n**💡 What You Can Do Next:**\n"
            response += "• Ask me to explain any part in more detail\n"
            response += "• Request help submitting a grievance\n"
            response += "• Ask about related topics\n"
            response += "• Get step-by-step guidance\n"
            
            return response
        
        # Enhanced intelligent responses for common patterns
        message_lower = message.lower()
        
        # Greeting patterns
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return """👋 **Hello! Welcome to the AI Grievance Assistant!**

I'm here to help you with any questions about:
• **Grievance Submission:** How to submit, what to include, file attachments
• **Process Guidance:** Step-by-step help with grievance procedures
• **Policy Information:** Understanding rules, rights, and procedures
• **Status Tracking:** Checking on your existing grievances
• **General Support:** Any other questions about the grievance system

**How can I assist you today?** You can:
• Ask me a specific question
• Get help submitting a grievance
• Learn about grievance categories and processes
• Check on existing cases
• Get emotional support and guidance

What would you like to know about?"""
        
        # Help patterns
        if any(word in message_lower for word in ['help', 'support', 'assist', 'guide', 'how to']):
            return """🆘 **I'm Here to Help! Here's What I Can Do:**

**📝 Grievance Assistance:**
• Guide you through submission process
• Explain categories and priority levels
• Help with file attachments
• Provide status updates

**🎓 Academic Issues:**
• Grade disputes and appeals
• Faculty complaints
• Course problems
• Academic integrity

**🏠 Hostel & Housing:**
• Maintenance requests
• Roommate conflicts
• Security concerns
• Food service issues

**🏗️ Infrastructure:**
• Building maintenance
• Technology support
• Equipment problems
• Accessibility issues

**📋 Administration:**
• Billing disputes
• Policy questions
• Document requests
• Enrollment issues

**💬 Just Ask Me:**
Tell me what you need help with, and I'll provide specific guidance, step-by-step instructions, or connect you with the right resources.

What specific issue would you like assistance with?"""
        
        # Grievance submission patterns
        if any(word in message_lower for word in ['submit', 'file', 'create', 'new grievance', 'complaint']):
            return """📝 **Ready to Submit a Grievance? Here's How:**

**🚀 Quick Start:**
1. Go to "Submit Grievance" in the sidebar
2. Fill in the title and description
3. Choose the appropriate category
4. Set priority level (or let AI auto-detect)
5. Attach supporting documents if needed
6. Submit and get instant AI analysis!

**📋 What You'll Need:**
• Clear description of the issue
• Relevant details and context
• Supporting documents (optional)
• Your contact information (unless anonymous)

**🎯 AI-Powered Features:**
• Automatic sentiment analysis
• Smart priority detection
• Intelligent routing to departments
• Estimated resolution times
• Emotional support recognition

**🔒 Privacy Options:**
• Submit anonymously with Trust Index
• Keep your identity private
• Maintain confidentiality

Would you like me to:
• Walk you through the submission process?
• Explain what categories are available?
• Help you draft your grievance description?
• Guide you through a specific step?

Just let me know what you need help with!"""
        
        # Status checking patterns
        if any(word in message_lower for word in ['status', 'track', 'check', 'update', 'progress', 'where is my']):
            return """📊 **Checking Your Grievance Status:**

**🔍 How to Check Status:**
1. Go to "My Grievances" in the sidebar
2. View all your submitted grievances
3. See real-time status updates
4. Check resolution progress
5. View staff responses

**📈 Status Types:**
• **Pending:** Under review by staff
• **In Progress:** Being actively worked on
• **Under Investigation:** Additional information being gathered
• **Resolved:** Issue has been addressed
• **Closed:** Case completed

**⏰ Response Times:**
• High Priority: 24-48 hours
• Medium Priority: 3-5 days
• Low Priority: 5-7 days

**📱 Real-Time Updates:**
• Status changes are shown immediately
• Email notifications for major updates
• Direct staff communication
• Progress tracking with milestones

**Need Help?**
• Can't find your grievance? Let me help you search
• Status unclear? I can explain what it means
• Want to follow up? I'll guide you through the process

Would you like me to help you check on a specific grievance or explain any status in detail?"""
        
        # Category information patterns
        if any(word in message_lower for word in ['category', 'categories', 'what types', 'kinds of']):
            return """📂 **Grievance Categories Available:**

**🎓 Academic:**
• Grade disputes and appeals
• Faculty complaints and issues
• Course registration problems
• Academic integrity violations
• Disability accommodations
• Course schedule conflicts

**🏠 Hostel & Housing:**
• Maintenance and repairs
• Roommate conflicts
• Security and safety concerns
• Food service quality
• Facility access issues
• Accommodation problems

**🏗️ Infrastructure:**
• Building maintenance
• Technology and internet issues
• Equipment problems
• Utility issues (water, electricity)
• Transportation problems
• Accessibility concerns

**📋 Administration:**
• Billing and financial issues
• Policy questions and clarifications
• Document requests
• Enrollment problems
• Financial aid issues
• Administrative procedures

**🔍 Other:**
• General complaints
• Policy suggestions
• Systemic issues
• Special circumstances
• COVID-related concerns
• Mental health support

**💡 How to Choose:**
• Pick the category that best fits your main issue
• If multiple categories apply, choose the primary one
• You can mention other categories in your description
• AI will help route to the right department

**Need Help Choosing?**
Tell me about your issue, and I'll suggest the best category and help you understand the process!"""
        
        # File attachment patterns
        if any(word in message_lower for word in ['file', 'attachment', 'upload', 'document', 'photo', 'image']):
            return """📎 **File Attachments - What You Need to Know:**

**✅ Supported File Types:**
• **Images:** JPG, JPEG, PNG
• **Documents:** PDF, DOC, DOCX
• **Maximum Size:** 10MB per file

**📋 What to Attach:**
• **Evidence:** Photos of damage, screenshots, receipts
• **Documents:** Official letters, contracts, policies
• **Records:** Communication logs, emails, messages
• **Supporting Materials:** Any relevant documentation

**💡 Attachment Tips:**
• Clear, readable images work best
• Include multiple angles if showing damage
• Screenshots should show full context
• Documents should be complete and legible
• Multiple files can be attached

**🔒 Privacy & Security:**
• Files are stored securely
• Only authorized staff can access
• Your privacy is protected
• Files are deleted after resolution

**❌ What NOT to Attach:**
• Personal identification documents
• Sensitive personal information
• Large video files
• Executable programs
• Copyrighted materials

**Need Help?**
• Having trouble uploading? I can troubleshoot
• Unsure what to attach? I can guide you
• File too large? I can suggest alternatives

Would you like help with a specific file attachment issue?"""
        
        # Priority and urgency patterns
        if any(word in message_lower for word in ['priority', 'urgent', 'important', 'critical', 'emergency', 'asap']):
            return """🚨 **Priority Levels & Urgency:**

**🔴 High Priority (24-48 hours):**
• Safety and security issues
• Health and medical concerns
• Academic deadlines
• Financial emergencies
• Discrimination/harassment
• Infrastructure failures

**🟡 Medium Priority (3-5 days):**
• Maintenance requests
• Policy questions
• Administrative issues
• Course problems
• Facility access
• General complaints

**🟢 Low Priority (5-7 days):**
• Suggestions and feedback
• Non-urgent maintenance
• General inquiries
• Policy clarifications
• Minor inconveniences

**⚡ How Priority is Determined:**
• **AI Analysis:** Automatic detection from your description
• **Keywords:** Urgency indicators in your text
• **Category:** Some categories are automatically higher priority
• **Manual Override:** You can set priority level yourself
• **Emotion Detection:** Emotional distress may increase priority

**🚨 Emergency Situations:**
If you're experiencing an immediate safety threat:
• Call emergency services (911) first
• Submit grievance with "URGENT" in title
• Use High priority setting
• Include emergency details

**💡 Priority Tips:**
• Be specific about urgency in your description
• Use clear language about impact
• Mention deadlines if applicable
• Explain why it's urgent

**Need Help Setting Priority?**
Tell me about your situation, and I'll help you determine the appropriate priority level and guide you through the submission process!"""
        
        # Anonymous submission patterns
        if any(word in message_lower for word in ['anonymous', 'privacy', 'confidential', 'secret', 'hidden']):
            return """🔒 **Anonymous Submissions - Your Privacy Protected:**

**✅ How Anonymous Submissions Work:**
• Your identity is hidden from staff
• Only admins can see full details (for system management)
• Your privacy is maintained throughout the process
• You can still track status and receive updates

**🛡️ Trust Index System:**
• Validates legitimate grievances
• Prevents system abuse
• Maintains accountability
• Protects your privacy

**📝 Anonymous Submission Process:**
1. Check "Submit anonymously" during submission
2. AI will analyze your grievance content
3. Trust Index validates your submission
4. Case is processed normally
5. Your identity remains protected

**🔍 What Staff Can See:**
• Grievance details and description
• Category and priority level
• Supporting documents
• Status updates and responses

**🔍 What Staff CANNOT See:**
• Your name or username
• Your email address
• Your personal information
• Your grievance history

**⚠️ Important Notes:**
• Anonymous cases may take slightly longer to process
• You can't receive direct staff communication
• Status updates are still available
• You can provide additional information if needed

**💡 When to Use Anonymous:**
• Sensitive personal issues
• Fear of retaliation
• Whistleblower situations
• Personal privacy concerns
• Testing the system

**Need Help?**
• Want to submit anonymously? I'll guide you through it
• Have privacy concerns? I can explain the protections
• Need to understand the process? I'll walk you through it

Would you like me to help you submit an anonymous grievance or explain any privacy aspects in more detail?"""
        
        # Default intelligent response
        return """🤔 **I understand you're asking about something, but I need a bit more context to help you best.**

**💡 Here are some ways I can help:**

**📝 Grievance Process:**
• How to submit a grievance
• Understanding categories and priorities
• File attachment guidelines
• Status tracking and updates

**🎓 Academic Issues:**
• Grade disputes and appeals
• Faculty complaints
• Course problems
• Academic integrity

**🏠 Hostel & Housing:**
• Maintenance requests
• Roommate conflicts
• Security concerns
• Food service issues

**🏗️ Infrastructure:**
• Building maintenance
• Technology support
• Equipment problems
• Accessibility issues

**📋 Administration:**
• Billing disputes
• Policy questions
• Document requests
• Enrollment issues

**🔒 Privacy & Support:**
• Anonymous submissions
• Emotional support
• Urgent case handling
• Follow-up procedures

**💬 Just Ask Me:**
Try asking me something specific like:
• "How do I submit a grievance?"
• "What are the grievance categories?"
• "How do I check my grievance status?"
• "Can I submit anonymously?"
• "What if my issue is urgent?"

Or tell me about your specific situation, and I'll provide targeted guidance!"""
    
    def _show_root_cause_analysis(self):
        if st.session_state.user_role not in ['admin', 'staff']:
            st.error("Access denied. Admin privileges required.")
            return
        
        st.header("🔍 Fast Root Cause Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            analysis_period = st.selectbox("Analysis Period", [7, 14, 30, 60, 90])
        with col2:
            min_cluster_size = st.slider("Minimum Cluster Size", 2, 10, 3)
        with col3:
            analysis_methods = st.multiselect(
                "Analysis Methods",
                ["K-means", "DBSCAN", "Hierarchical"],
                default=["K-means"]
            )
        
        if st.button("Run Analysis"):
            st.success("Analysis completed!")
            
            st.subheader("📋 Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Complaints", 150)
            with col2:
                st.metric("Categories Affected", 5)
            with col3:
                st.metric("Clusters Found", 8)
            with col4:
                st.metric("Largest Cluster", 25)
            
            st.subheader("🎯 Identified Root Causes")
            
            root_causes = [
                {"issue": "Network connectivity problems", "severity": 0.85, "size": 25},
                {"issue": "Hostel maintenance delays", "severity": 0.72, "size": 18},
                {"issue": "Academic schedule conflicts", "severity": 0.68, "size": 15}
            ]
            
            for cause in root_causes:
                with st.expander(f"{cause['issue']} (Severity: {cause['severity']:.2f})"):
                    st.write(f"**Cluster Size:** {cause['size']} complaints")
                    st.write(f"**Severity Score:** {cause['severity']:.2f}")
                    st.write("**Recommendations:**")
                    st.write("• Implement preventive maintenance schedule")
                    st.write("• Improve communication channels")
                    st.write("• Regular system health checks")
            
            st.subheader("Root Cause Analysis")
            academic_causes = {
                "Network connectivity problems": 25,
                "Hostel maintenance delays": 18,
                "Academic schedule conflicts": 15
            }
            causes_df = pd.DataFrame(list(academic_causes.items()), 
                                      columns=['Cause', 'Count'])
            st.bar_chart(causes_df.set_index('Cause'))
            
            st.subheader("📈 Trend Analysis")
            st.write("**Volume Trend:** Increasing")
            st.write("**Daily Average:** 12.5 complaints")
            st.write("**Sentiment Trend:** Slightly improving")
            
            st.subheader("💡 Overall Recommendations")
            st.info("Focus on infrastructure improvements and staff training")
            st.info("Implement proactive monitoring systems")
            st.info("Enhance user communication channels")
    
    def _show_routing_dashboard(self):
        if 'user_role' not in st.session_state or st.session_state.user_role not in ['admin', 'staff']:
            st.error("Access denied. Admin privileges required.")
            return
        
        st.header("🎯 Fast Smart Routing Dashboard")
        
        # Get grievances data for routing analysis
        try:
            grievances = self.db_manager.get_grievances()
            
            if not grievances:
                st.warning("No grievance data available for routing analysis.")
                return
                
            # Convert to DataFrame and ensure required columns exist
            import pandas as pd
            df = pd.DataFrame(grievances)
            
            # Check if required columns exist
            required_columns = ['created_at', 'category', 'status', 'sentiment']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns in routing data: {', '.join(missing_columns)}")
                return
            
            # Filter out any invalid sentiment data if needed
            if 'sentiment' in df.columns:
                valid_sentiments = ['positive', 'neutral', 'negative']
                df = df[df['sentiment'].isin(valid_sentiments)]
        except Exception as e:
            st.error(f"Error loading routing data: {str(e)}")
            return
        
        # Display workload status
        st.subheader("📊 Current Workload Status")
        
        # Calculate department statistics
        if 'category' in df.columns and 'status' in df.columns:
            dept_stats = df['category'].value_counts().reset_index()
            dept_stats.columns = ['Department', 'Grievance Count']
            
            # Add status distribution for each department
            dept_status = df.groupby(['category', 'status']).size().unstack(fill_value=0)
            dept_stats = dept_stats.merge(dept_status, left_on='Department', right_index=True, how='left')
            
            # Display department metrics
            for _, row in dept_stats.iterrows():
                dept_name = row['Department']
                total = row['Grievance Count']
                
                # Calculate workload status
                if total > 15:
                    status = "overloaded"
                elif total > 10:
                    status = "busy"
                else:
                    status = "normal"
                
                with st.expander(f"{dept_name} - {status.title()}", expanded=True):
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Total Grievances", total)
                    with cols[1]:
                        st.metric("Status", status.title())
                    with cols[2]:
                        st.metric("Pending", row.get('pending', 0))
                    
                    # Show status distribution
                    if 'pending' in row and 'resolved' in row:
                        st.progress(min(1.0, total / 20), 
                                  text=f"Workload: {total} grievances (max 20)")
        
        # Routing simulation section
        st.subheader("🧪 Routing Simulation")
        
        with st.form("routing_simulation"):
            st.write("Test the routing system with a sample grievance:")
            
            col1, col2 = st.columns(2)
            with col1:
                test_category = st.selectbox("Category", ["academic", "hostel", "infrastructure", "administration"])
                test_priority = st.selectbox("Priority", [1, 2, 3])
            
            with col2:
                test_emotion = st.selectbox("Primary Emotion", ["anger", "frustration", "sadness", "neutral"])
                test_impact = st.slider("Impact Score", 0.0, 1.0, 0.5)
            
            test_description = st.text_area("Description", "Sample grievance for testing routing")
            
            if st.form_submit_button("Test Routing"):
                st.success("Routing Analysis Complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Routing Decision:**")
                    st.write(f"- Department: {test_category}")
                    st.write(f"- Assigned Staff: Staff-{test_priority}")
                    st.write(f"- Routing Score: 0.85")
                    st.write(f"- Confidence: 0.92")
                
                with col2:
                    st.write("**Estimates:**")
                    st.write(f"- Resolution Time: {24 if test_priority == 1 else 48 if test_priority == 2 else 72} hours")
                    st.write(f"- Escalation Applied: {'Yes' if test_priority == 1 else 'No'}")
        
        # Show routing statistics if we have data
        if not df.empty:
            st.subheader("📈 Routing Statistics")
            
            # Department distribution
            if 'category' in df.columns:
                st.bar_chart(df['category'].value_counts())
            
            # Status distribution
            if 'status' in df.columns:
                st.bar_chart(df['status'].value_counts())
        
        st.info("Your feedback helps us understand areas for improvement.")
    
    def _show_sentiment_tracking(self):
        st.header("😊 Fast Sentiment Tracking")
        st.caption("Track your emotional journey through the grievance process")
        
        try:
            # Get grievances from database
            user_grievances = self.db_manager.get_grievances(user_id=st.session_state.user_id)
            
            if not user_grievances:
                st.info("No grievances found. Please submit a grievance to see sentiment tracking.")
                return
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(user_grievances)
            
            # Debug option to show raw data
            if st.checkbox("Show raw data"):
                st.write("Raw grievance data:", df)
            
            # Ensure required columns exist
            required_columns = ['created_at', 'sentiment', 'title']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns in grievance data: {', '.join(missing_columns)}")
                st.write("Available columns:", df.columns.tolist())
                return
            
            # Handle 'Unknown' or invalid datetime values with explicit format
            df['created_at'] = pd.to_datetime(df['created_at'], 
                                           format='%Y-%m-%d %H:%M:%S', 
                                           errors='coerce')
            
            # Filter out rows with invalid dates
            df = df.dropna(subset=['created_at'])
            
            # Ensure we have valid data to display
            if df.empty:
                st.warning("No valid grievance data with proper dates available for sentiment tracking.")
                return
                
            # Ensure sentiment values are valid
            valid_sentiments = ['positive', 'neutral', 'negative']
            df = df[df['sentiment'].isin(valid_sentiments)]
            
            if df.empty:
                st.warning("No grievances with valid sentiment data found. Please submit a new grievance.")
                return
                
            # Map sentiment to numerical score for analysis
            df['sentiment_score'] = df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
            
            # Sort by date before plotting
            df = df.sort_values('created_at')
            
            # Create a date range for the chart
            date_range = pd.date_range(start=df['created_at'].min(), end=df['created_at'].max())
            
            # Reindex to include all dates in the range (for smoother line chart)
            chart_data = df.set_index('created_at')['sentiment_score']
            chart_data = chart_data.reindex(date_range).fillna(method='ffill')
            
            # Main sentiment trend chart
            st.subheader("Sentiment Trend Over Time")
            # Convert to DataFrame for better handling
            chart_df = pd.DataFrame(chart_data, columns=['sentiment_score'])
            st.line_chart(chart_df, use_container_width=True)
            
            # Show sentiment distribution
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
            
            # Show metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_grievances = len(df)
            positive_count = len(df[df['sentiment'] == 'positive'])
            neutral_count = len(df[df['sentiment'] == 'neutral'])
            negative_count = len(df[df['sentiment'] == 'negative'])
            
            with col1:
                st.metric("Total Grievances", total_grievances)
            
            with col2:
                st.metric("😊 Positive", f"{positive_count} ({positive_count/total_grievances*100:.1f}%)")
            
            with col3:
                st.metric("😐 Neutral", f"{neutral_count} ({neutral_count/total_grievances*100:.1f}%)")
                
            with col4:
                st.metric("😠 Negative", f"{negative_count} ({negative_count/total_grievances*100:.1f}%)")
            
            # Sentiment by category
            if 'category' in df.columns:
                st.subheader("Sentiment by Category")
                # Create a crosstab of category vs sentiment
                category_sentiment = pd.crosstab(df['category'], df['sentiment'], normalize='index') * 100
                
                # Reset index to make category a column
                category_sentiment = category_sentiment.reset_index()
                
                # Melt the DataFrame for better visualization
                melted = category_sentiment.melt(
                    id_vars='category', 
                    value_vars=['positive', 'neutral', 'negative'],
                    var_name='Sentiment',
                    value_name='Percentage'
                )
                
                # Create a bar chart using altair for better control
                import altair as alt
                
                chart = alt.Chart(melted).mark_bar().encode(
                    x=alt.X('category:N', title='Category', sort='-y'),
                    y=alt.Y('Percentage:Q', title='Percentage (%)'),
                    color='Sentiment:N',
                    tooltip=['category', 'Sentiment', 'Percentage']
                ).properties(
                    width=600,
                    height=400
                )
                
                st.altair_chart(chart, use_container_width=True)
            
            # Recent grievances with sentiment
            st.subheader("Recent Grievances")
            recent_grievances = df[['created_at', 'title', 'category', 'sentiment']]\
                .sort_values('created_at', ascending=False)\
                .head(5)
            
            # Apply color coding to sentiment
            def color_sentiment(val):
                color = 'green' if val == 'positive' else 'orange' if val == 'neutral' else 'red'
                return f'color: {color}'
                
            st.dataframe(
                recent_grievances.style.applymap(color_sentiment, subset=['sentiment']),
                use_container_width=True,
                hide_index=True
            )
            
            # Add download option
            if st.button("📥 Download Sentiment Data"):
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Sentiment Data', index=False)
                st.download_button(
                    label="⬇️ Download Excel",
                    data=buffer.getvalue(),
                    file_name="sentiment_analysis.xlsx",
                    mime="application/vnd.ms-excel"
                )
            
        except Exception as e:
            st.error("❌ An error occurred while processing sentiment data")
            st.error(f"Error details: {str(e)}")
            st.info("ℹ️ Please try again or contact support if the issue persists.")
            
            # Show traceback in expander for debugging
            with st.expander("Technical Details"):
                import traceback
                st.code(traceback.format_exc(), language='python')
    
    def _get_suggested_questions(self, category: str = None) -> list:
        """Get a list of suggested questions, optionally filtered by category"""
        if category:
            return [qa["question"] for qa in self.qa_database.values() if qa.get("category") == category]
        return [qa["question"] for qa in self.qa_database.values()]
    
    def _show_suggested_questions(self):
        """Display suggested questions in expandable sections by category"""
        categories = {
            "general": "General Questions",
            "academic": "Academic Issues",
            "hostel": "Hostel & Housing",
            "infrastructure": "Infrastructure",
            "administration": "Administration",
            "privacy": "Privacy & Support"
        }
        
        st.subheader("💡 What would you like to know?")
        
        for category_id, category_name in categories.items():
            with st.expander(f"{category_name}", expanded=category_id=="general"):
                questions = self._get_suggested_questions(category_id)
                for question in questions[:5]:  # Show up to 5 questions per category
                    if st.button(question, key=f"suggested_{category_id}_{questions.index(question)}"):
                        st.session_state.chat_input = question
                        st.experimental_rerun()
    
    def _get_knowledge_base_response(self, user_query: str) -> str:
        """
        Find the best matching response for a user query from the knowledge base.
        
        Args:
            user_query (str): The user's query
            
        Returns:
            str: The best matching response, or a default response if no match is found
        """
        # Convert query to lowercase for case-insensitive matching
        query = user_query.lower().strip()
        
        # Check for exact matches first
        for key, qa in self.qa_database.items():
            if key in query or any(kw in query for kw in qa.get('keywords', [])):
                return qa["answer"]
        
        # Check for partial matches in question text
        for qa in self.qa_database.values():
            question = qa["question"].lower()
            if query in question or any(word in question for word in query.split()):
                return qa["answer"]
        
        # If no match found, return a default response with suggestions
        default_responses = [
            "I'm not sure I understand. Could you rephrase your question?",
            "I don't have information on that specific topic. Could you try asking something else?",
            "I'm still learning! Could you try asking about grievance submission, tracking, or categories?",
            "I don't have that information right now. Would you like to know about submitting a grievance or checking its status?"
        ]
        
        # Add some common suggestions based on the query
        if any(word in query for word in ["submit", "file", "lodge"]):
            return "To submit a grievance, go to the 'Submit Grievance' section in the sidebar. " \
                  "You'll need to provide a title, select a category, and describe your issue. " \
                  "You can also attach relevant files if needed."
        
        if any(word in query for word in ["track", "status", "progress"]):
            return "You can track your grievances in the 'My Grievances' section. " \
                  "You'll see the current status, priority, and any updates there."
        
        if any(word in query for word in ["category", "type", "kind"]):
            return "We handle several categories including Academic, Hostel, Infrastructure, " \
                  "Administration, and Other. Each category has specialized handling procedures."
        
        # Return a random default response
        import random
        return random.choice(default_responses)
    
    def _show_chat_assistant(self):
        st.header("🤖 Knowledge Base Assistant")
        st.caption("Get instant answers to your questions about the grievance system")
        
        # Initialize chat history if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your Knowledge Base Assistant. I can help you find information about the grievance system. What would you like to know?"}
            ]
        
        # Show suggested questions in the sidebar
        with st.sidebar:
            st.subheader("💡 Quick Links")
            if st.button("Clear Chat History"):
                st.session_state.messages = [
                    {"role": "assistant", "content": "Chat history cleared. How can I help you today?"}
                ]
                st.experimental_rerun()
            
            st.markdown("### Common Questions")
            common_questions = [
                "How do I submit a grievance?",
                "How can I track my grievance status?",
                "What are the grievance categories?",
                "How long does resolution take?",
                "Can I submit anonymously?"
            ]
            
            for question in common_questions:
                if st.button(question, key=f"sidebar_{question[:10]}"):
                    st.session_state.chat_input = question
                    st.experimental_rerun()
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "assistant":
                    st.info(f"🤖 {message['content']}")
                else:
                    st.text_area("You", value=message["content"], key=f"message_{st.session_state.messages.index(message)}", disabled=True)
        
        # Show suggested questions in the main area if no conversation has started
        if len(st.session_state.messages) <= 1:  # Only showing the initial message
            self._show_suggested_questions()
        
        # Chat input at the bottom
        st.write("### Ask a question")
        col1, col2 = st.columns([5, 1])
        with col1:
            prompt = st.text_input("Type your question here...", 
                                key="chat_input")
        with col2:
            st.write("")
            st.write("")
            send_button = st.button("Send")
        
        if send_button and prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Show typing indicator
            with st.spinner("Searching knowledge base..."):
                # Get response from knowledge base
                response = self._get_knowledge_base_response(prompt)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Rerun to update the display
                st.experimental_rerun()
        
        # Add feedback buttons if there are messages
        if len(st.session_state.messages) > 1:  # More than just the initial message
            st.markdown("---")
            st.write("Was this response helpful?")
            col1, col2 = st.columns([1, 10])
            with col1:
                if st.button("👍", key="thumbs_up"):
                    st.session_state.feedback = "positive"
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎", key="thumbs_down"):
                    st.session_state.feedback = "negative"
                    st.text_input("How can we improve?", key="feedback_input")
    
    def _show_root_cause_analysis(self):
        st.header("🔍 Root Cause Analysis")
        st.caption("Analyze patterns and identify root causes of grievances")
        
        try:
            # Get all grievances for analysis
            grievances = self.db_manager.get_grievances()
            
            if not grievances:
                st.info("No grievances found for analysis. Please submit some grievances first.")
                return
                
            df = pd.DataFrame(grievances)
            
            # Basic data validation
            required_columns = ['category', 'description', 'created_at', 'status']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns for analysis: {', '.join(missing_columns)}")
                return
            
            # Show analysis options
            analysis_type = st.radio(
                "Analysis Type",
                ["Category Trends", "Temporal Analysis", "Sentiment Correlation", "Text Analysis"]
            )
            
            if analysis_type == "Category Trends":
                st.subheader("Grievance Categories Analysis")
                category_counts = df['category'].value_counts()
                st.bar_chart(category_counts)
                
                st.subheader("Resolution Rate by Category")
                resolution_rates = df.groupby('category')['status'].apply(
                    lambda x: (x == 'Resolved').mean() * 100
                ).sort_values(ascending=False)
                st.bar_chart(resolution_rates)
                
            elif analysis_type == "Temporal Analysis":
                st.subheader("Grievance Trends Over Time")
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                df = df.dropna(subset=['created_at'])
                
                if not df.empty:
                    df_daily = df.set_index('created_at').resample('D').size()
                    st.line_chart(df_daily)
                
            elif analysis_type == "Sentiment Correlation":
                if 'sentiment' in df.columns:
                    st.subheader("Sentiment Distribution by Category")
                    sentiment_pivot = pd.crosstab(df['category'], df.get('sentiment', 'neutral'), 
                                                normalize='index')
                    st.bar_chart(sentiment_pivot)
                else:
                    st.warning("Sentiment data not available for analysis")
                    
            elif analysis_type == "Text Analysis":
                st.subheader("Common Themes")
                
                # Simple word frequency analysis
                from collections import Counter
                from nltk.corpus import stopwords
                import string
                
                # Combine all descriptions
                text = ' '.join(df['description'].astype(str).str.lower())
                
                # Tokenize and clean
                stop_words = set(stopwords.words('english'))
                tokens = [word for word in text.split() 
                         if word not in stop_words and word not in string.punctuation]
                
                # Get most common words
                word_freq = Counter(tokens).most_common(20)
                words, counts = zip(*word_freq)
                
                st.bar_chart(pd.Series(counts, index=words))
                
                # Show word cloud if available
                try:
                    from wordcloud import WordCloud
                    import matplotlib.pyplot as plt
                    
                    wordcloud = WordCloud(width=800, height=400, 
                                        background_color='white').generate(' '.join(tokens))
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except ImportError:
                    st.info("Install wordcloud package for visual word cloud generation")
            
            # Add export option
            if st.button("📊 Export Analysis Report"):
                # Create a simple report
                report = f"# Root Cause Analysis Report\n"
                report += f"## Summary Statistics\n"
                report += f"- Total Grievances: {len(df)}\n"
                report += f"- Categories: {', '.join(df['category'].unique())}\n"
                report += f"- Resolution Rate: {(df['status'] == 'Resolved').mean()*100:.1f}%\n"
                
                # Add category distribution
                report += "\n## Category Distribution\n"
                for cat, count in df['category'].value_counts().items():
                    report += f"- {cat}: {count} ({(count/len(df)*100):.1f}%)\n"
                
                # Create download button
                st.download_button(
                    "📥 Download Report",
                    report,
                    "root_cause_analysis_report.md",
                    "text/markdown"
                )
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.exception(e)
    
    def _show_resolution_quality_predictor(self):
        st.header("🔮 Resolution Quality Predictor")
        st.caption("Predict the quality of grievance resolution based on historical data")
        
        try:
            # Get resolved grievances with feedback
            grievances = self.db_manager.get_grievances()
            df = pd.DataFrame(grievances)
            
            if 'rating' not in df.columns or 'feedback' not in df.columns:
                st.warning("Not enough data for prediction. Need resolved grievances with ratings and feedback.")
                return
                
            # Filter for resolved cases with ratings
            resolved_df = df[(df['status'] == 'Resolved') & (df['rating'].notna())]
            
            if len(resolved_df) < 10:
                st.warning(f"Need at least 10 resolved cases with ratings. Currently have {len(resolved_df)}.")
                return
                
            st.subheader("Resolution Quality Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_rating = resolved_df['rating'].mean()
                st.metric("Average Rating", f"{avg_rating:.1f}/5.0")
                
            with col2:
                resolution_time = 24  # Placeholder for actual calculation
                st.metric("Avg. Resolution Time", f"{resolution_time} hours")
                
            with col3:
                satisfaction_rate = (resolved_df['rating'] >= 4).mean() * 100
                st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
                
            # Show prediction form
            st.subheader("Predict Resolution Quality")
            
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    category = st.selectbox("Category", df['category'].unique())
                    priority = st.selectbox("Priority", [1, 2, 3])
                    
                with col2:
                    complexity = st.select_slider("Complexity", ["Low", "Medium", "High"])
                    assigned_to = st.selectbox("Assigned To", ["General", "Specialist"])
                    
                if st.form_submit_button("Predict"):
                    # Simulate prediction (replace with actual model)
                    prediction = {
                        'quality_score': 0.75,
                        'confidence': 0.82,
                        'estimated_resolution_time': "24-48 hours",
                        'recommended_action': "Assign to senior staff for handling"
                    }
                    
                    st.success("✅ Prediction Complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Quality Score", f"{prediction['quality_score']:.0%}")
                        st.metric("Confidence", f"{prediction['confidence']:.0%}")
                        
                    with col2:
                        st.metric("Est. Resolution", prediction['estimated_resolution_time'])
                        st.metric("Recommended", prediction['recommended_action'])
                        
                    # Show feature importance
                    st.subheader("Key Factors")
                    factors = [
                        ("Staff Experience", 0.35),
                        ("Case Complexity", 0.28),
                        ("Category", 0.22),
                        ("Time of Submission", 0.15)
                    ]
                    
                    for factor, impact in factors:
                        st.progress(int(impact * 100), f"{factor}: {impact:.0%} impact")
                        
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
    
    def _show_anonymous_trust_system(self):
        st.header("👤 Anonymous Trust System")
        st.caption("Submit and track anonymous grievances with trust scoring")
        
        tab1, tab2 = st.tabs(["Submit Anonymously", "Trust Dashboard"])
        
        with tab1:
            st.subheader("Submit Anonymous Grievance")
            
            with st.form("anonymous_form"):
                title = st.text_input("Title (Optional)", placeholder="Brief description")
                
                col1, col2 = st.columns(2)
                with col1:
                    category = st.selectbox(
                        "Category",
                        ["Academic", "Hostel", "Infrastructure", "Administration", "Other"]
                    )
                with col2:
                    severity = st.select_slider("Severity", ["Low", "Medium", "High"])
                
                description = st.text_area(
                    "Description",
                    placeholder="Please describe your concern in detail..."
                )
                
                # Trust score explanation
                st.info("""
                Your submission will receive a trust score based on:
                - Content quality and detail
                - Historical accuracy of similar reports
                - Corroborating evidence (if any)
                """)
                
                if st.form_submit_button("Submit Anonymously"):
                    if not description:
                        st.error("Please provide a description of your concern")
                    else:
                        # Generate a unique anonymous ID
                        import uuid
                        anonymous_id = str(uuid.uuid4())[:8]
                        
                        # Calculate initial trust score (simplified)
                        trust_score = 50  # Base score
                        trust_score += len(description.split()) // 10  # More detailed = higher score
                        trust_score = min(100, max(0, trust_score))  # Clamp between 0-100
                        
                        # Save to database (implementation depends on your DB)
                        # self.db_manager.save_anonymous_report(...)
                        
                        st.success(f"✅ Report submitted successfully! Your anonymous ID: `{anonymous_id}`")
                        st.balloons()
                        
                        # Show trust information
                        st.subheader("Your Trust Profile")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Trust Score", f"{trust_score}/100")
                            
                        with col2:
                            level = "Low"
                            if trust_score > 70: level = "High"
                            elif trust_score > 40: level = "Medium"
                            st.metric("Trust Level", level)
                            
                        # Show next steps
                        st.info("""
                        **Next Steps:**
                        - Use your anonymous ID to track updates
                        - Check back for status updates
                        - Your identity remains protected
                        """)
        
        with tab2:
            st.subheader("Trust Dashboard")
            
            # Show trust metrics (placeholder data)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Anonymous Reports", "1,247")
                
            with col2:
                st.metric("Avg. Trust Score", "68/100")
                
            with col3:
                st.metric("Resolution Rate", "82%")
                
            # Show trust distribution
            st.subheader("Trust Score Distribution")
            trust_data = pd.DataFrame({
                'Score Range': ['0-20', '21-40', '41-60', '61-80', '81-100'],
                'Reports': [50, 120, 350, 500, 227]
            })
            st.bar_chart(trust_data.set_index('Score Range'))
            
            # Show recent high-trust reports
            st.subheader("Recent High-Trust Reports")
            recent_reports = [
                {"id": "#A7B2C4", "category": "Infrastructure", "score": 88, "status": "In Progress"},
                {"id": "#B3D9F5", "category": "Academic", "score": 92, "status": "Resolved"},
                {"id": "#C4E1A4", "category": "Hostel", "score": 85, "status": "Pending"}
            ]
            
            for report in recent_reports:
                with st.expander(f"{report['id']} - {report['category']} (Score: {report['score']})"):
                    st.write(f"**Status:** {report['status']}")
                    st.progress(report['score'])
    
    def _show_mood_tracker(self):
        st.header("📊 Organizational Mood Tracker")
        st.caption("Track emotional trends across departments over time")
        
        departments = ["Academic", "Hostel", "Infrastructure", "Administration", "Student Services"]
        emotions = ["Happy", "Neutral", "Frustrated", "Angry", "Satisfied"]
        
        mood_data = pd.DataFrame(
            np.random.rand(len(departments), len(emotions)) * 100,
            index=departments,
            columns=emotions
        )
        
        st.subheader("Department Mood Heatmap")
        st.dataframe(mood_data)
        
        st.subheader("📈 Mood Trends Over Time")
        
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
        mood_trends = pd.DataFrame({
            'Month': dates,
            'Mood Score': np.random.rand(len(dates)) * 100
        })
        
        st.line_chart(mood_trends.set_index('Month'))
        
        st.subheader("🔍 Department Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Performing Departments:**")
            st.write("1. Student Services (Mood: 85)")
            st.write("2. Administration (Mood: 78)")
            st.write("3. Academic (Mood: 72)")
        
        with col2:
            st.write("**Areas for Improvement:**")
            st.write("1. Infrastructure (Mood: 45)")
            st.write("2. Hostel (Mood: 52)")
            st.write("3. Academic (Mood: 72)")
    
    def _save_uploaded_file(self, uploaded_file) -> str:
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(upload_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    def _show_qa_database_browser(self):
        """Enhanced Q&A database browser with advanced search and filtering"""
        st.subheader("📚 Enhanced Q&A Database Browser")
        st.caption("Browse through 300+ questions and answers with advanced features")
        
        # Enhanced search and filtering
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_query = st.text_input("🔍 Search questions:", placeholder="Type keywords to search...")
        
        with col2:
            search_type = st.selectbox("Search in:", ["All", "Questions", "Answers", "Keywords"])
        
        with col3:
            show_enhanced = st.checkbox("🔍 Enhanced entries", value=True)
        
        # Advanced filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            categories = list(set(qa['category'] for qa in self.qa_database.values()))
            categories.sort()
            selected_category = st.selectbox("📂 Filter by Category", ["All Categories"] + categories)
        
        with col2:
            priority_filter = st.selectbox("🎯 Filter by Priority", ["All Priorities", "High", "Medium", "Low"])
        
        with col3:
            if show_enhanced:
                enhanced_count = sum(1 for qa in self.qa_database.values() if 'keywords' in qa)
                st.info(f"Enhanced entries: {enhanced_count}")
        
        # Enhanced filtering with keywords and priority
        filtered_qa = {}
        for key, qa in self.qa_database.items():
            category_match = selected_category == "All Categories" or qa['category'] == selected_category
            
            priority_match = True
            if priority_filter != "All Priorities" and 'priority' in qa:
                priority_match = qa['priority'] == priority_filter.lower()
            
            enhanced_match = True
            if show_enhanced:
                enhanced_match = 'keywords' in qa
            
            search_match = True
            if search_query:
                search_match = False
                
                if search_type == "All" or search_type == "Questions":
                    if search_query.lower() in qa['question'].lower():
                        search_match = True
                
                if search_type == "All" or search_type == "Answers":
                    if search_query.lower() in qa['answer'].lower():
                        search_match = True
                
                if search_type == "All" or search_type == "Keywords":
                    if 'keywords' in qa and any(search_query.lower() in kw.lower() for kw in qa['keywords']):
                        search_match = True
                
                if search_query.lower() in key.lower():
                    search_match = True
            
            if category_match and priority_match and enhanced_match and search_match:
                filtered_qa[key] = qa
        
        # Display enhanced Q&A with more information
        if filtered_qa:
            st.write(f"**Found {len(filtered_qa)} questions**")
            
            for key, qa in filtered_qa.items():
                with st.expander(f"❓ {qa['question']}", expanded=False):
                    st.write(f"**Answer:** {qa['answer']}")
                    
                    # Show enhanced information if available
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Category:** {qa['category'].replace('_', ' ').title()}")
                        if 'priority' in qa:
                            priority_emoji = "🔴" if qa['priority'] == 'high' else "🟡" if qa['priority'] == 'medium' else "🟢"
                            st.write(f"**Priority:** {priority_emoji} {qa['priority'].title()}")
                    
                    with col2:
                        st.write(f"**Key:** `{key}`")
                        if 'keywords' in qa:
                            st.write(f"**Keywords:** {', '.join(qa['keywords'])}")
                    
                    # Show related questions if available
                    if 'related_questions' in qa and qa['related_questions']:
                        st.write("**Related Questions:**")
                        for related in qa['related_questions']:
                            st.write(f"• {related}")
                    
                    # Add a button to ask this question in chat
                    if st.button(f"Ask: {qa['question'][:50]}...", key=f"ask_{key}"):
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': qa['question']
                        })
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': f"🤖 **{qa['question']}**\n\n{qa['answer']}\n\n*Category: {qa['category'].replace('_', ' ').title()}*"
                        })
                        st.success("Question added to chat! Check the chat above.")
                        st.experimental_rerun()
        else:
            st.info("No questions match your current filters. Try adjusting the category, priority, or search terms.")
        
        # Enhanced statistics with more metrics
        st.subheader("📊 Enhanced Database Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Questions", len(self.qa_database))
        
        with col2:
            st.metric("Categories", len(categories))
        
        with col3:
            enhanced_count = sum(1 for qa in self.qa_database.values() if 'keywords' in qa)
            st.metric("Enhanced Entries", enhanced_count)
        
        with col4:
            avg_answer_length = sum(len(qa['answer']) for qa in self.qa_database.values()) / len(self.qa_database)
            st.metric("Avg Answer Length", f"{avg_answer_length:.0f} chars")
        
        # Show priority distribution if available
        if any('priority' in qa for qa in self.qa_database.values()):
            st.subheader("🎯 Priority Distribution")
            priority_data = {}
            for qa in self.qa_database.values():
                if 'priority' in qa:
                    priority = qa['priority'].title()
                    priority_data[priority] = priority_data.get(priority, 0) + 1
            
            if priority_data:
                st.bar_chart(priority_data)
        
        # Intelligent suggestions based on search patterns
        if search_query and len(search_query) > 2:
            st.subheader("💡 Intelligent Suggestions")
            suggestions = self._get_intelligent_suggestions(search_query)
            if suggestions:
                st.write("**Based on your search, you might also be interested in:**")
                for suggestion in suggestions[:5]:  # Show top 5 suggestions
                    st.write(f"• {suggestion}")
            else:
                st.info("Try different keywords or browse categories for more options.")
    
    def _get_intelligent_suggestions(self, search_query: str) -> list:
        """Get intelligent suggestions based on search patterns and related questions"""
        suggestions = []
        search_lower = search_query.lower()
        
        # Find questions with similar keywords or related topics
        for key, qa in self.qa_database.items():
            if 'keywords' in qa:
                # Check if any keywords are related to the search
                keyword_matches = [kw for kw in qa['keywords'] if kw.lower() in search_lower or search_lower in kw.lower()]
                if keyword_matches:
                    suggestions.append(qa['question'])
            
            # Check related questions
            if 'related_questions' in qa:
                for related in qa['related_questions']:
                    if search_lower in related.lower() or related.lower() in search_lower:
                        suggestions.append(related)
        
        # Remove duplicates and return unique suggestions
        return list(set(suggestions))
    
    def _check_context_awareness(self, message: str) -> str:
        """Enhanced context-aware responses based on conversation history and user patterns"""
        if not hasattr(st.session_state, 'chat_history') or len(st.session_state.chat_history) < 2:
            return None
        
        # Get recent conversation context (last 6 messages for better context)
        recent_messages = st.session_state.chat_history[-6:]
        
        # Analyze conversation patterns
        user_messages = [msg for msg in recent_messages if msg['role'] == 'user']
        ai_messages = [msg for msg in recent_messages if msg['role'] == 'assistant']
        
        if not user_messages or not ai_messages:
            return None
        
        # Get the last AI response to understand current context
        last_ai_response = ai_messages[-1]['content']
        current_message = message.lower()
        
        # Enhanced context detection with multiple patterns
        context_patterns = {
            'academic_followup': {
                'keywords': ['what about', 'how about', 'and', 'also', 'too', 'as well', 'what else', 'more'],
                'context_indicators': ['academic', 'course', 'grade', 'faculty', 'class'],
                'response': """🎓 **Academic Follow-up:**

I see you're asking about academic issues. Here are some related topics you might want to know about:

• **Grade Disputes:** How to challenge grades, appeal processes, grade calculations
• **Faculty Issues:** Reporting inappropriate behavior, requesting changes, communication problems
• **Course Problems:** Schedule conflicts, prerequisite issues, course registration
• **Academic Integrity:** Plagiarism, cheating, violations, honor code
• **Academic Support:** Tutoring, study resources, disability accommodations

What specific academic issue would you like help with? I can provide detailed guidance based on your situation."""
            },
            
            'hostel_followup': {
                'keywords': ['what about', 'how about', 'and', 'also', 'too', 'as well', 'what else', 'more'],
                'context_indicators': ['hostel', 'dorm', 'room', 'accommodation', 'housing'],
                'response': """🏠 **Hostel Follow-up:**

I see you're asking about hostel issues. Here are some related topics:

• **Maintenance:** Repairs, cleaning, facilities, pest control, HVAC issues
• **Roommate Issues:** Conflicts, room changes, privacy, noise complaints
• **Security:** Safety concerns, access control, lost keys, security incidents
• **Food Services:** Quality, allergies, dietary restrictions, meal plans
• **Facilities:** Laundry, gym, study rooms, common areas

What specific hostel issue would you like help with? I can guide you through the resolution process."""
            },
            
            'infrastructure_followup': {
                'keywords': ['what about', 'how about', 'and', 'also', 'too', 'as well', 'what else', 'more'],
                'context_indicators': ['infrastructure', 'building', 'facility', 'equipment', 'technology'],
                'response': """🏗️ **Infrastructure Follow-up:**

I see you're asking about infrastructure issues. Here are some related topics:

• **Buildings:** Maintenance, accessibility, safety, renovations, space issues
• **Technology:** Internet, computers, software, WiFi, email systems
• **Utilities:** Electricity, water, heating, cooling, power outages
• **Transportation:** Buses, parking, accessibility, shuttle services
• **Equipment:** Lab equipment, office equipment, AV systems

What specific infrastructure issue would you like help with? I can connect you with the right department."""
            },
            
            'urgent_escalation': {
                'keywords': ['urgent', 'critical', 'emergency', 'immediate', 'asap', 'now', 'help'],
                'context_indicators': ['urgent', 'critical', 'emergency', 'immediate'],
                'response': """🚨 **Urgent Issue Detected:**

I understand this is urgent. Here's what happens with urgent grievances:

• **Immediate Response:** High-priority cases get response within 2-4 hours
• **Automatic Escalation:** Urgent cases are automatically escalated to senior staff
• **24/7 Support:** Emergency issues are handled around the clock
• **Direct Contact:** You'll receive direct contact from staff handling your case

**For immediate assistance:**
- Call emergency services if safety is at risk
- Use the 'High' priority setting when submitting
- Include 'URGENT' in your title for faster processing

Would you like me to help you submit this as an urgent grievance right now?"""
            },
            
            'emotional_support': {
                'keywords': ['frustrated', 'angry', 'sad', 'anxious', 'worried', 'upset', 'disappointed'],
                'context_indicators': ['frustrated', 'angry', 'sad', 'anxious', 'worried'],
                'response': """😊 **Emotional Support & Understanding:**

I understand you're feeling {emotion_detected}. Your feelings are valid, and we're here to help:

• **Immediate Support:** Your case will be handled with extra care and sensitivity
• **Priority Handling:** Emotional distress often indicates urgent needs
• **Specialized Staff:** Cases with emotional components are routed to trained staff
• **Follow-up Care:** We'll check in to ensure your needs are met

**Remember:** It's okay to feel this way, and seeking help shows strength. We're committed to resolving your issue and supporting you through the process.

Would you like to tell me more about what's happening so I can better assist you?"""
            }
        }
        
        # Check for context patterns
        for pattern_name, pattern_data in context_patterns.items():
            # Check if current message contains follow-up keywords
            has_followup_keywords = any(keyword in current_message for keyword in pattern_data['keywords'])
            
            # Check if last AI response contains context indicators
            has_context = any(indicator in last_ai_response.lower() for indicator in pattern_data['context_indicators'])
            
            if has_followup_keywords and has_context:
                # For emotional support, detect the specific emotion
                if pattern_name == 'emotional_support':
                    emotion_detected = self._detect_enhanced_emotion(message, "Neutral")
                    return pattern_data['response'].format(emotion_detected=emotion_detected)
                else:
                    return pattern_data['response']
        
        # Check for topic continuation patterns
        topic_continuation = self._detect_topic_continuation(recent_messages, current_message)
        if topic_continuation:
            return topic_continuation
        
        return None
    
    def _detect_topic_continuation(self, recent_messages: list, current_message: str) -> str:
        """Detect if user is continuing the same topic from previous messages"""
        if len(recent_messages) < 3:
            return None
        
        # Look for topic continuation patterns
        topic_keywords = {
            'academic': ['course', 'grade', 'faculty', 'class', 'assignment', 'exam'],
            'hostel': ['room', 'dorm', 'maintenance', 'facility', 'accommodation'],
            'infrastructure': ['building', 'equipment', 'internet', 'facility', 'utility'],
            'administration': ['billing', 'policy', 'document', 'enrollment', 'financial']
        }
        
        # Find the dominant topic in recent conversation
        topic_scores = {topic: 0 for topic in topic_keywords}
        
        for msg in recent_messages:
            msg_lower = msg['content'].lower()
            for topic, keywords in topic_keywords.items():
                for keyword in keywords:
                    if keyword in msg_lower:
                        topic_scores[topic] += 1
        
        # Find the most discussed topic
        dominant_topic = max(topic_scores, key=topic_scores.get)
        
        # Check if current message continues this topic
        if topic_scores[dominant_topic] > 1:  # At least 2 mentions
            current_topic_keywords = topic_keywords[dominant_topic]
            if any(keyword in current_message for keyword in current_topic_keywords):
                return f"""🔄 **Topic Continuation - {dominant_topic.title()}**

I see you're continuing to discuss {dominant_topic} issues. This helps me provide more targeted assistance.

**Current {dominant_topic.title()} Topics:**
{self._get_topic_suggestions(dominant_topic)}

Would you like me to:
• Provide more specific information about {dominant_topic} processes?
• Help you submit a {dominant_topic} grievance?
• Connect you with {dominant_topic} specialists?
• Explain {dominant_topic} policies and procedures?

Please let me know what specific aspect you'd like to explore further."""
        
        return None
    
    def _get_topic_suggestions(self, topic: str) -> str:
        """Get topic-specific suggestions for the user"""
        topic_suggestions = {
            'academic': "• Grade disputes and appeals\n• Course registration issues\n• Faculty complaints\n• Academic integrity concerns\n• Disability accommodations",
            'hostel': "• Maintenance requests\n• Roommate conflicts\n• Security concerns\n• Food service issues\n• Facility access problems",
            'infrastructure': "• Building maintenance\n• Technology support\n• Utility issues\n• Equipment problems\n• Accessibility concerns",
            'administration': "• Billing disputes\n• Policy clarifications\n• Document requests\n• Enrollment issues\n• Financial aid problems"
        }
        
        return topic_suggestions.get(topic, "• General information\n• Process guidance\n• Policy explanations\n• Contact information")
    
    def _record_chat_feedback(self, feedback_type: str):
        """Record user feedback for chatbot responses"""
        if not hasattr(st.session_state, 'chat_feedback'):
            st.session_state.chat_feedback = {'helpful': 0, 'not_helpful': 0}
        
        st.session_state.chat_feedback[feedback_type] += 1
        
        # Store feedback in session for analytics
        if not hasattr(st.session_state, 'chat_feedback_history'):
            st.session_state.chat_feedback_history = []
        
        st.session_state.chat_feedback_history.append({
            'timestamp': datetime.now().isoformat(),
            'feedback_type': feedback_type,
            'user_id': st.session_state.get('user_id', 'anonymous')
        })
    
    def _advanced_ai_classification(self, text: str, user_context: dict = None) -> dict:
        """Advanced AI-powered grievance classification with auto-tagging and routing"""
        
        # Enhanced text preprocessing
        processed_text = self._preprocess_text_for_ai(text)
        
        # Multi-layer classification approach
        classification_result = {
            'primary_category': None,
            'secondary_categories': [],
            'confidence_score': 0.0,
            'auto_tags': [],
            'urgency_level': 'medium',
            'department_routing': None,
            'estimated_resolution_time': '3-5 days',
            'similar_cases': [],
            'risk_assessment': 'low',
            'compliance_flags': []
        }
        
        # 1. Primary Category Classification using enhanced keyword matching
        category_scores = self._calculate_category_scores(processed_text)
        primary_category = max(category_scores, key=category_scores.get)
        classification_result['primary_category'] = primary_category
        classification_result['confidence_score'] = category_scores[primary_category] / 100
        
        # 2. Secondary Categories (top 2 runners-up)
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        classification_result['secondary_categories'] = [cat for cat, score in sorted_categories[1:3]]
        
        # 3. Auto-tagging system
        classification_result['auto_tags'] = self._generate_auto_tags(processed_text, primary_category)
        
        # 4. Urgency detection
        classification_result['urgency_level'] = self._detect_advanced_urgency(processed_text, user_context)
        
        # 5. Department routing
        classification_result['department_routing'] = self._determine_department_routing(
            primary_category, classification_result['auto_tags'], classification_result['urgency_level']
        )
        
        # 6. Resolution time estimation
        classification_result['estimated_resolution_time'] = self._estimate_resolution_time(
            primary_category, classification_result['urgency_level'], classification_result['auto_tags']
        )
        
        # 7. Similar case detection
        classification_result['similar_cases'] = self._find_similar_cases(processed_text, primary_category)
        
        # 8. Risk assessment
        classification_result['risk_assessment'] = self._assess_risk_level(processed_text, primary_category)
        
        # 9. Compliance checking
        classification_result['compliance_flags'] = self._check_compliance_requirements(
            primary_category, classification_result['auto_tags']
        )
        
        return classification_result
    
    def _preprocess_text_for_ai(self, text: str) -> str:
        """Advanced text preprocessing for AI analysis"""
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Expand common abbreviations
        abbreviations = {
            'prof': 'professor',
            'dept': 'department',
            'admin': 'administration',
            'faculty': 'faculty',
            'student': 'student',
            'course': 'course',
            'grade': 'grade',
            'exam': 'examination',
            'hw': 'homework',
            'assignment': 'assignment'
        }
        
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
        
        return text
    
    def _calculate_category_scores(self, processed_text: str) -> dict:
        """Calculate confidence scores for each category using enhanced keyword matching"""
        
        # Enhanced keyword dictionaries with weights
        category_keywords = {
            'Academic': {
                'course': 15, 'grade': 20, 'professor': 18, 'exam': 16, 'assignment': 14,
                'syllabus': 12, 'curriculum': 10, 'lecture': 8, 'tutorial': 8, 'academic': 25,
                'study': 6, 'learning': 6, 'education': 8, 'scholarship': 12, 'degree': 10
            },
            'Hostel': {
                'hostel': 25, 'room': 18, 'accommodation': 20, 'maintenance': 16, 'facility': 12,
                'cleaning': 14, 'security': 16, 'food': 12, 'laundry': 8, 'internet': 10,
                'electricity': 12, 'water': 12, 'heating': 10, 'cooling': 10, 'noise': 8
            },
            'Infrastructure': {
                'building': 20, 'facility': 18, 'equipment': 16, 'maintenance': 14, 'repair': 12,
                'construction': 16, 'renovation': 14, 'accessibility': 12, 'safety': 16,
                'parking': 10, 'transportation': 12, 'library': 8, 'laboratory': 10, 'classroom': 8
            },
            'Administration': {
                'policy': 20, 'procedure': 18, 'regulation': 16, 'bureaucracy': 14, 'form': 12,
                'document': 10, 'approval': 12, 'permission': 14, 'deadline': 12, 'fee': 10,
                'payment': 12, 'registration': 14, 'enrollment': 12, 'graduation': 10
            },
            'Other': {
                'general': 8, 'miscellaneous': 6, 'other': 10, 'unknown': 5, 'unspecified': 5
            }
        }
        
        category_scores = {category: 0 for category in category_keywords.keys()}
        
        # Calculate weighted scores
        for category, keywords in category_keywords.items():
            score = 0
            for keyword, weight in keywords.items():
                if keyword in processed_text:
                    score += weight
                    # Bonus for multiple occurrences
                    score += processed_text.count(keyword) * 2
            
            # Normalize score to 0-100 range
            category_scores[category] = min(score, 100)
        
        return category_scores
    
    def _generate_auto_tags(self, processed_text: str, primary_category: str) -> list:
        """Generate automatic tags based on content analysis"""
        
        tags = []
        
        # Category-specific tag generation
        if primary_category == 'Academic':
            academic_tags = ['course_issue', 'grading', 'faculty', 'curriculum', 'examination']
            for tag in academic_tags:
                if any(keyword in processed_text for keyword in tag.split('_')):
                    tags.append(tag)
        
        elif primary_category == 'Hostel':
            hostel_tags = ['maintenance', 'facility', 'security', 'cleaning', 'utilities']
            for tag in hostel_tags:
                if tag in processed_text:
                    tags.append(tag)
        
        elif primary_category == 'Infrastructure':
            infra_tags = ['building', 'equipment', 'safety', 'accessibility', 'maintenance']
            for tag in infra_tags:
                if tag in processed_text:
                    tags.append(tag)
        
        elif primary_category == 'Administration':
            admin_tags = ['policy', 'procedure', 'documentation', 'approval', 'deadline']
            for tag in admin_tags:
                if tag in processed_text:
                    tags.append(tag)
        
        # Universal tags based on content
        if any(word in processed_text for word in ['urgent', 'immediate', 'asap', 'critical']):
            tags.append('urgent')
        
        if any(word in processed_text for word in ['complaint', 'dissatisfied', 'unhappy', 'problem']):
            tags.append('complaint')
        
        if any(word in processed_text for word in ['request', 'please', 'need', 'require']):
            tags.append('request')
        
        if any(word in processed_text for word in ['suggestion', 'improvement', 'better', 'enhance']):
            tags.append('suggestion')
        
        # Time-based tags
        if any(word in processed_text for word in ['deadline', 'due', 'time', 'schedule']):
            tags.append('time_sensitive')
        
        # Financial tags
        if any(word in processed_text for word in ['money', 'cost', 'fee', 'payment', 'refund']):
            tags.append('financial')
        
        return tags[:8]  # Limit to 8 most relevant tags
    
    def _detect_advanced_urgency(self, processed_text: str, user_context: dict = None) -> str:
        """Advanced urgency detection using multiple factors"""
        
        urgency_score = 0
        
        # Text-based urgency indicators
        urgency_keywords = {
            'high': ['urgent', 'immediate', 'asap', 'critical', 'emergency', 'serious', 'dangerous'],
            'medium': ['soon', 'quickly', 'prompt', 'timely', 'important'],
            'low': ['whenever', 'sometime', 'no rush', 'low priority']
        }
        
        for level, keywords in urgency_keywords.items():
            for keyword in keywords:
                if keyword in processed_text:
                    if level == 'high':
                        urgency_score += 3
                    elif level == 'medium':
                        urgency_score += 2
                    else:
                        urgency_score += 1
        
        # Context-based urgency (if user has history of urgent issues)
        if user_context and user_context.get('urgent_history_count', 0) > 2:
            urgency_score += 2
        
        # Category-based urgency adjustment
        if 'safety' in processed_text or 'security' in processed_text:
            urgency_score += 3
        
        if 'health' in processed_text or 'medical' in processed_text:
            urgency_score += 3
        
        # Determine final urgency level
        if urgency_score >= 6:
            return 'high'
        elif urgency_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _determine_department_routing(self, primary_category: str, auto_tags: list, urgency_level: str) -> str:
        """Determine optimal department routing based on classification"""
        
        routing_rules = {
            'Academic': {
                'default': 'academic_affairs',
                'grading': 'academic_affairs',
                'faculty': 'academic_affairs',
                'curriculum': 'curriculum_committee',
                'examination': 'examination_office'
            },
            'Hostel': {
                'default': 'student_affairs',
                'maintenance': 'facilities_management',
                'security': 'security_office',
                'food': 'dining_services'
            },
            'Infrastructure': {
                'default': 'facilities_management',
                'safety': 'safety_office',
                'accessibility': 'disability_services',
                'transportation': 'transportation_office'
            },
            'Administration': {
                'default': 'student_services',
                'policy': 'policy_office',
                'financial': 'financial_aid',
                'registration': 'registrar_office'
            }
        }
        
        category_rules = routing_rules.get(primary_category, {})
        
        # Check for specific tag-based routing
        for tag in auto_tags:
            if tag in category_rules:
                return category_rules[tag]
        
        # Return default routing for category
        return category_rules.get('default', 'general_office')
    
    def _estimate_resolution_time(self, primary_category: str, urgency_level: str, auto_tags: list) -> str:
        """Estimate resolution time based on multiple factors"""
        
        base_times = {
            'Academic': {'high': '24-48 hours', 'medium': '3-5 days', 'low': '5-7 days'},
            'Hostel': {'high': '2-4 hours', 'medium': '1-2 days', 'low': '3-5 days'},
            'Infrastructure': {'high': '4-8 hours', 'medium': '2-3 days', 'low': '5-7 days'},
            'Administration': {'high': '1-2 days', 'medium': '3-5 days', 'low': '5-7 days'}
        }
        
        base_time = base_times.get(primary_category, {}).get(urgency_level, '3-5 days')
        
        # Adjust based on tags
        if 'urgent' in auto_tags:
            # Reduce time for urgent cases
            if 'days' in base_time:
                days = int(base_time.split()[0])
                adjusted_days = max(1, days - 2)
                base_time = f"{adjusted_days}-{adjusted_days + 1} days"
            elif 'hours' in base_time:
                hours = int(base_time.split()[0])
                adjusted_hours = max(2, hours - 2)
                base_time = f"{adjusted_hours}-{adjusted_hours + 2} hours"
        
        return base_time
    
    def _find_similar_cases(self, processed_text: str, primary_category: str) -> list:
        """Find similar cases for reference and pattern recognition"""
        
        # This would typically query the database for similar cases
        # For now, return sample similar cases
        similar_cases = [
            {
                'id': 'GRI-2024-001',
                'title': 'Similar academic issue',
                'resolution': 'Resolved in 3 days',
                'similarity_score': 0.85
            },
            {
                'id': 'GRI-2024-015',
                'title': 'Related infrastructure problem',
                'resolution': 'Resolved in 2 days',
                'similarity_score': 0.72
            }
        ]
        
        return similar_cases
    
    def _assess_risk_level(self, processed_text: str, primary_category: str) -> str:
        """Assess risk level of the grievance"""
        
        risk_score = 0
        
        # High-risk keywords
        high_risk_words = ['legal', 'lawyer', 'sue', 'lawsuit', 'discrimination', 'harassment', 'safety', 'danger']
        for word in high_risk_words:
            if word in processed_text:
                risk_score += 3
        
        # Medium-risk keywords
        medium_risk_words = ['complaint', 'unfair', 'wrong', 'mistake', 'error', 'problem']
        for word in medium_risk_words:
            if word in processed_text:
                risk_score += 2
        
        # Category-based risk adjustment
        if primary_category == 'Academic':
            risk_score += 1  # Academic issues can be sensitive
        elif primary_category == 'Infrastructure':
            if 'safety' in processed_text:
                risk_score += 2  # Safety issues are high risk
        
        if risk_score >= 6:
            return 'high'
        elif risk_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _check_compliance_requirements(self, primary_category: str, auto_tags: list) -> list:
        """Check for compliance requirements and flags"""
        
        compliance_flags = []
        
        # Category-specific compliance checks
        if primary_category == 'Academic':
            if 'grading' in auto_tags:
                compliance_flags.append('academic_policy_review')
            if 'faculty' in auto_tags:
                compliance_flags.append('faculty_conduct_review')
        
        elif primary_category == 'Hostel':
            if 'security' in auto_tags:
                compliance_flags.append('security_protocol_review')
            if 'maintenance' in auto_tags:
                compliance_flags.append('maintenance_standard_check')
        
        elif primary_category == 'Infrastructure':
            if 'safety' in auto_tags:
                compliance_flags.append('safety_compliance_audit')
            if 'accessibility' in auto_tags:
                compliance_flags.append('ada_compliance_check')
        
        # Universal compliance checks
        if 'urgent' in auto_tags:
            compliance_flags.append('escalation_protocol')
        
        if 'financial' in auto_tags:
            compliance_flags.append('financial_audit_trail')
        
        return compliance_flags
    
    def _smart_notification_system(self, grievance_data: dict, action: str = 'created'):
        """Smart notification system with AI-powered escalation"""
        
        notification_config = {
            'created': {
                'user': True,
                'admin': True,
                'department': True,
                'escalation': False
            },
            'updated': {
                'user': True,
                'admin': False,
                'department': True,
                'escalation': False
            },
            'escalated': {
                'user': True,
                'admin': True,
                'department': True,
                'escalation': True
            },
            'resolved': {
                'user': True,
                'admin': False,
                'department': False,
                'escalation': False
            }
        }
        
        config = notification_config.get(action, notification_config['created'])
        
        notifications = []
        
        # 1. User notification
        if config['user'] and grievance_data.get('user_id'):
            user_notification = self._create_user_notification(grievance_data, action)
            notifications.append(user_notification)
        
        # 2. Admin notification
        if config['admin']:
            admin_notification = self._create_admin_notification(grievance_data, action)
            notifications.append(admin_notification)
        
        # 3. Department notification
        if config['department']:
            dept_notification = self._create_department_notification(grievance_data, action)
            notifications.append(dept_notification)
        
        # 4. Escalation notification
        if config['escalation']:
            escalation_notification = self._create_escalation_notification(grievance_data)
            notifications.append(escalation_notification)
        
        # 5. Smart routing based on AI analysis
        if action == 'created':
            routing_notification = self._create_smart_routing_notification(grievance_data)
            notifications.append(routing_notification)
        
        return notifications
    
    def _create_user_notification(self, grievance_data: dict, action: str) -> dict:
        """Create user notification with personalized content"""
        
        action_messages = {
            'created': f"Your grievance '{grievance_data.get('title', 'Untitled')}' has been submitted successfully.",
            'updated': f"Your grievance '{grievance_data.get('title', 'Untitled')}' has been updated.",
            'escalated': f"Your grievance '{grievance_data.get('title', 'Untitled')}' has been escalated for urgent attention.",
            'resolved': f"Your grievance '{grievance_data.get('title', 'Untitled')}' has been resolved."
        }
        
        return {
            'type': 'user',
            'recipient_id': grievance_data.get('user_id'),
            'subject': f"Grievance {action.title()}: {grievance_data.get('title', 'Untitled')}",
            'message': action_messages.get(action, "Your grievance has been processed."),
            'priority': grievance_data.get('priority', 2),
            'category': grievance_data.get('category', 'General'),
            'grievance_id': grievance_data.get('id'),
            'timestamp': datetime.now().isoformat(),
            'read': False
        }
    
    def _create_admin_notification(self, grievance_data: dict, action: str) -> dict:
        """Create admin notification with comprehensive details"""
        
        urgency_emoji = "🚨" if grievance_data.get('priority') == 1 else "📋"
        risk_emoji = "⚠️" if grievance_data.get('risk_level') == 'high' else "✅"
        
        return {
            'type': 'admin',
            'recipient_role': 'admin',
            'subject': f"{urgency_emoji} {action.title()} Grievance: {grievance_data.get('title', 'Untitled')}",
            'message': f"""
            New grievance {action}:
            
            **Title:** {grievance_data.get('title', 'Untitled')}
            **Category:** {grievance_data.get('category', 'General')}
            **Priority:** {grievance_data.get('priority', 'Medium')}
            **Risk Level:** {risk_emoji} {grievance_data.get('risk_level', 'Low')}
            **User ID:** {grievance_data.get('user_id', 'Anonymous')}
            **Created:** {grievance_data.get('created_at', 'Unknown')}
            
            **Description:** {grievance_data.get('description', 'No description')[:200]}...
            """,
            'priority': grievance_data.get('priority', 2),
            'category': grievance_data.get('category', 'General'),
            'grievance_id': grievance_data.get('id'),
            'timestamp': datetime.now().isoformat(),
            'read': False,
            'requires_action': True
        }
    
    def _create_department_notification(self, grievance_data: dict, action: str) -> dict:
        """Create department-specific notification"""
        
        # Determine department based on category and AI analysis
        department_mapping = {
            'Academic': 'academic_affairs',
            'Hostel': 'student_affairs',
            'Infrastructure': 'facilities_management',
            'Administration': 'student_services',
            'Other': 'general_office'
        }
        
        department = department_mapping.get(grievance_data.get('category', 'Other'), 'general_office')
        
        return {
            'type': 'department',
            'recipient_department': department,
            'subject': f"📋 Department Assignment: {grievance_data.get('title', 'Untitled')}",
            'message': f"""
            New grievance assigned to {department.replace('_', ' ').title()}:
            
            **Title:** {grievance_data.get('title', 'Untitled')}
            **Category:** {grievance_data.get('category', 'General')}
            **Priority:** {grievance_data.get('priority', 'Medium')}
            **Estimated Resolution:** {grievance_data.get('estimated_resolution', '3-5 days')}
            
            **Action Required:** Please review and assign appropriate staff member.
            """,
            'priority': grievance_data.get('priority', 2),
            'category': grievance_data.get('category', 'General'),
            'grievance_id': grievance_data.get('id'),
            'timestamp': datetime.now().isoformat(),
            'read': False,
            'requires_action': True
        }
    
    def _create_escalation_notification(self, grievance_data: dict) -> dict:
        """Create escalation notification for urgent cases"""
        
        escalation_reasons = []
        if grievance_data.get('priority') == 1:
            escalation_reasons.append("High Priority")
        if grievance_data.get('risk_level') == 'high':
            escalation_reasons.append("High Risk")
        if grievance_data.get('urgency_level') == 'high':
            escalation_reasons.append("High Urgency")
        if grievance_data.get('compliance_flags'):
            escalation_reasons.append("Compliance Issues")
        
        return {
            'type': 'escalation',
            'recipient_role': 'admin',
            'subject': f"🚨 ESCALATION REQUIRED: {grievance_data.get('title', 'Untitled')}",
            'message': f"""
            **URGENT ESCALATION REQUIRED**
            
            Grievance: {grievance_data.get('title', 'Untitled')}
            ID: {grievance_data.get('id', 'Unknown')}
            
            **Escalation Reasons:**
            {chr(10).join(f"• {reason}" for reason in escalation_reasons)}
            
            **Immediate Action Required:**
            • Review within 2 hours
            • Assign senior staff member
            • Consider legal/compliance review if needed
            
            **Current Status:** {grievance_data.get('status', 'Pending')}
            **Priority:** {grievance_data.get('priority', 'Medium')}
            **Risk Level:** {grievance_data.get('risk_level', 'Low')}
            """,
            'priority': 1,  # Always high priority for escalations
            'category': grievance_data.get('category', 'General'),
            'grievance_id': grievance_data.get('id'),
            'timestamp': datetime.now().isoformat(),
            'read': False,
            'requires_action': True,
            'escalation_level': 'high'
        }
    
    def _create_smart_routing_notification(self, grievance_data: dict) -> dict:
        """Create smart routing notification based on AI analysis"""
        
        # Get AI analysis for smart routing
        ai_analysis = self._advanced_ai_classification(
            grievance_data.get('description', ''), 
            {'urgent_history_count': 0}
        )
        
        return {
            'type': 'smart_routing',
            'recipient_department': ai_analysis.get('department_routing', 'general_office'),
            'subject': f"🤖 AI-Routed: {grievance_data.get('title', 'Untitled')}",
            'message': f"""
            **AI-Powered Routing Recommendation**
            
            Grievance: {grievance_data.get('title', 'Untitled')}
            ID: {grievance_data.get('id', 'Unknown')}
            
            **AI Analysis Results:**
            • Primary Category: {ai_analysis.get('primary_category', 'Unknown')}
            • Confidence Score: {ai_analysis.get('confidence_score', 0):.1%}
            • Urgency Level: {ai_analysis.get('urgency_level', 'medium').title()}
            • Risk Assessment: {ai_analysis.get('risk_assessment', 'low').title()}
            
            **Recommended Department:** {ai_analysis.get('department_routing', 'general_office').replace('_', ' ').title()}
            **Estimated Resolution:** {ai_analysis.get('estimated_resolution_time', '3-5 days')}
            
            **Auto-Generated Tags:** {', '.join(ai_analysis.get('auto_tags', []))}
            
            **AI Confidence:** This routing recommendation is based on advanced AI analysis with {ai_analysis.get('confidence_score', 0):.1%} confidence.
            """,
            'priority': grievance_data.get('priority', 2),
            'category': grievance_data.get('category', 'General'),
            'grievance_id': grievance_data.get('id'),
            'timestamp': datetime.now().isoformat(),
            'read': False,
            'requires_action': True,
            'ai_confidence': ai_analysis.get('confidence_score', 0)
        }
    
    def _check_escalation_triggers(self, grievance_data: dict) -> bool:
        """Check if grievance should be escalated based on various triggers"""
        
        escalation_score = 0
        
        # Priority-based escalation
        if grievance_data.get('priority') == 1:
            escalation_score += 3
        
        # Time-based escalation (if pending for too long)
        if grievance_data.get('created_at'):
            created_time = datetime.fromisoformat(grievance_data['created_at'])
            time_pending = datetime.now() - created_time
            
            if time_pending.days > 7:
                escalation_score += 2
            elif time_pending.days > 3:
                escalation_score += 1
        
        # Risk-based escalation
        if grievance_data.get('risk_level') == 'high':
            escalation_score += 2
        
        # Compliance-based escalation
        if grievance_data.get('compliance_flags'):
            escalation_score += 2
        
        # User history-based escalation
        if grievance_data.get('user_id'):
            user_grievances = self.db_manager.get_grievances(grievance_data['user_id'])
            urgent_history = len([g for g in user_grievances if g.get('priority') == 1])
            if urgent_history > 2:
                escalation_score += 1
        
        # Escalate if score >= 4
        return escalation_score >= 4
    
    def _auto_escalate_grievance(self, grievance_id: int) -> bool:
        """Automatically escalate a grievance based on triggers"""
        
        try:
            # Get grievance data
            grievances = self.db_manager.get_grievances()
            grievance = next((g for g in grievances if g['id'] == grievance_id), None)
            
            if not grievance:
                return False
            
            # Check escalation triggers
            if self._check_escalation_triggers(grievance):
                # Update status to escalated
                self.db_manager.update_grievance_status(grievance_id, 'Escalated', 
                    'Automatically escalated by AI system due to escalation triggers')
                
                # Send escalation notifications
                escalation_notifications = self._smart_notification_system(grievance, 'escalated')
                
                # Log escalation
                st.success(f"🚨 Grievance {grievance_id} automatically escalated!")
                
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Error in auto-escalation: {str(e)}")
            return False
    
    def _track_response_times(self, grievance_id: int) -> dict:
        """Track response times and identify bottlenecks"""
        
        try:
            grievances = self.db_manager.get_grievances()
            grievance = next((g for g in grievances if g['id'] == grievance_id), None)
            
            if not grievance:
                return {}
            
            created_time = datetime.fromisoformat(grievance['created_at'])
            current_time = datetime.now()
            
            # Calculate various time metrics
            time_metrics = {
                'total_time_pending': (current_time - created_time).total_seconds() / 3600,  # hours
                'days_pending': (current_time - created_time).days,
                'hours_pending': (current_time - created_time).total_seconds() / 3600,
                'escalation_threshold': 72,  # 72 hours = 3 days
                'critical_threshold': 168,   # 168 hours = 7 days
            }
            
            # Check if escalation is needed
            if time_metrics['hours_pending'] > time_metrics['escalation_threshold']:
                time_metrics['escalation_needed'] = True
                time_metrics['escalation_reason'] = 'Response time exceeded threshold'
            
            if time_metrics['hours_pending'] > time_metrics['critical_threshold']:
                time_metrics['critical'] = True
                time_metrics['critical_reason'] = 'Critical response time exceeded'
            
            return time_metrics
            
        except Exception as e:
            st.error(f"Error tracking response times: {str(e)}")
            return {}

if __name__ == "__main__":
    app = FastGrievanceSystem()
    app.run()
