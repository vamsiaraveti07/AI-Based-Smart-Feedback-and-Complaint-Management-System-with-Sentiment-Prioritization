"""
Emotion-Aware Chatbot
Advanced chatbot with emotional understanding and escalation capabilities
"""
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import sqlite3
import logging
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import random
from collections import defaultdict, deque
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Try to download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class EmotionAwareChatbot:
    """
    Advanced chatbot with emotional intelligence and escalation capabilities
    """
    
    def __init__(self, db_path: str = 'grievance_system.db'):
        self.db_path = db_path
        
        # Initialize sentiment analyzers
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize conversation state tracking
        self.conversation_history = defaultdict(lambda: deque(maxlen=10))
        self.user_emotional_state = defaultdict(dict)
        self.escalation_triggers = defaultdict(int)
        
        # Initialize response templates and emotional models
        self._init_emotional_models()
        self._init_response_templates()
        self._init_escalation_rules()
        
        # Create database tables
        self._create_chatbot_tables()
        
        # Load conversation history
        self._load_conversation_history()
    
    def _init_emotional_models(self):
        """Initialize emotion detection and response models"""
        try:
            # Emotion classification model
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Text generation model for empathetic responses
            self.text_generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Emotional models initialized successfully")
        except Exception as e:
            logger.warning(f"Could not load advanced models: {e}")
            self.emotion_classifier = None
            self.text_generator = None
    
    def _init_response_templates(self):
        """Initialize emotional response templates"""
        self.emotional_responses = {
            'anger': {
                'acknowledgment': [
                    "I understand you're feeling frustrated, and that's completely valid.",
                    "I can sense your anger, and I want to help resolve this issue for you.",
                    "Your frustration is justified, and I'm here to make things right.",
                    "I hear your anger, and I'm committed to finding a solution together."
                ],
                'empathy': [
                    "I can imagine how upsetting this situation must be for you.",
                    "Anyone would be angry in your situation.",
                    "This is clearly causing you significant stress.",
                    "I understand why you're feeling this way."
                ],
                'action': [
                    "Let me escalate this to our senior team immediately.",
                    "I'm going to prioritize your case right away.",
                    "I'll ensure this gets urgent attention.",
                    "This requires immediate action, and I'll make sure it happens."
                ]
            },
            'sadness': {
                'acknowledgment': [
                    "I can hear the disappointment in your message.",
                    "I understand this situation is causing you distress.",
                    "I recognize how disheartening this must be.",
                    "I can sense your sadness about this issue."
                ],
                'empathy': [
                    "I'm truly sorry you're going through this.",
                    "This must be really difficult for you.",
                    "I wish this hadn't happened to you.",
                    "I understand how disappointing this must be."
                ],
                'action': [
                    "Let me personally ensure this gets proper attention.",
                    "I'll monitor your case closely to ensure progress.",
                    "I want to make sure we turn this situation around for you.",
                    "I'm going to personally follow up on this."
                ]
            },
            'anxiety': {
                'acknowledgment': [
                    "I can sense your concern about this issue.",
                    "I understand you're worried about how this will be resolved.",
                    "I recognize your anxiety about this situation.",
                    "I can tell this is causing you stress."
                ],
                'empathy': [
                    "It's natural to feel anxious about this.",
                    "Your concerns are completely understandable.",
                    "Anyone would be worried in this situation.",
                    "I understand why this is causing you anxiety."
                ],
                'action': [
                    "Let me give you a clear timeline for resolution.",
                    "I'll provide you with regular updates to ease your concerns.",
                    "I'll make sure you're kept informed every step of the way.",
                    "Let me outline exactly what will happen next."
                ]
            },
            'frustration': {
                'acknowledgment': [
                    "I can tell this situation is really frustrating for you.",
                    "I understand your frustration with this ongoing issue.",
                    "I recognize how exasperating this must be.",
                    "I can sense your irritation, and it's completely justified."
                ],
                'empathy': [
                    "I would be frustrated too in your position.",
                    "This kind of situation would annoy anyone.",
                    "Your frustration is completely reasonable.",
                    "I understand why this is so irritating."
                ],
                'action': [
                    "Let me cut through the red tape and get this resolved.",
                    "I'll make sure this gets expedited attention.",
                    "I'm going to streamline the process for you.",
                    "Let me personally handle this to avoid further delays."
                ]
            },
            'neutral': {
                'acknowledgment': [
                    "Thank you for bringing this to our attention.",
                    "I understand you have a concern you'd like addressed.",
                    "I'm here to help you with your query.",
                    "Let me assist you with this matter."
                ],
                'empathy': [
                    "I appreciate you taking the time to contact us.",
                    "Your feedback is important to us.",
                    "I'm glad you reached out to us about this.",
                    "Thank you for giving us the opportunity to help."
                ],
                'action': [
                    "I'll make sure this gets proper attention.",
                    "Let me look into this for you.",
                    "I'll ensure this is handled appropriately.",
                    "I'll investigate this matter thoroughly."
                ]
            },
            'joy': {
                'acknowledgment': [
                    "I'm glad to hear from you!",
                    "It's wonderful that you're reaching out.",
                    "I appreciate your positive approach.",
                    "Thank you for your constructive feedback."
                ],
                'empathy': [
                    "I'm happy to assist you today.",
                    "It's great to work with someone so positive.",
                    "Your enthusiasm is appreciated.",
                    "I'm pleased to help you with this."
                ],
                'action': [
                    "I'll be happy to help you with this.",
                    "I'll make sure this gets the attention it deserves.",
                    "Let me assist you in the best way possible.",
                    "I'll ensure we handle this well for you."
                ]
            }
        }
        
        # FAQ responses
        self.faq_responses = {
            'status': [
                "To check your complaint status, please provide your grievance ID or I can look it up with your details.",
                "I can help you track your complaint. Could you share your grievance reference number?",
                "Let me check the status of your complaint. What's your grievance ID?"
            ],
            'submit': [
                "I'd be happy to help you submit a new grievance. What category does your concern fall under?",
                "I can guide you through submitting a complaint. What type of issue are you experiencing?",
                "Let me assist you with submitting a new grievance. Could you tell me more about your concern?"
            ],
            'timeline': [
                "Response times vary by priority and complexity. High-priority issues are typically resolved within 24-48 hours.",
                "Most complaints are acknowledged within 24 hours and resolved within 3-7 days depending on complexity.",
                "We aim to resolve urgent matters within 24-48 hours, and standard issues within a week."
            ],
            'escalation': [
                "If you're not satisfied with the response, I can escalate your case to a supervisor immediately.",
                "I can escalate this to our senior team if you feel it needs higher-level attention.",
                "Would you like me to escalate this to management for faster resolution?"
            ]
        }
        
        # Emotional escalation phrases
        self.escalation_phrases = {
            'high_intensity': [
                'furious', 'livid', 'outraged', 'disgusted', 'appalled', 'infuriated',
                'absolutely unacceptable', 'completely ridiculous', 'utterly disappointed'
            ],
            'repeated_frustration': [
                'again', 'still not fixed', 'same problem', 'how many times',
                'keep happening', 'repeatedly', 'constantly'
            ],
            'urgent_indicators': [
                'urgent', 'emergency', 'immediate', 'critical', 'asap', 'right now',
                'cannot wait', 'time sensitive'
            ],
            'threat_indicators': [
                'complain to', 'report to', 'take legal action', 'contact media',
                'social media', 'lawyer', 'sue', 'legal notice'
            ]
        }
    
    def _init_escalation_rules(self):
        """Initialize escalation rules and thresholds"""
        self.escalation_thresholds = {
            'emotion_intensity': 0.8,  # Above this level, consider escalation
            'consecutive_negative': 3,  # 3 consecutive negative interactions
            'frustration_keywords': 2,  # 2 or more frustration keywords
            'conversation_length': 5,   # 5+ exchanges without resolution
            'threat_detected': 1,       # Any threat indicator triggers escalation
            'sentiment_deterioration': -0.5  # Sentiment drop threshold
        }
        
        self.escalation_actions = {
            'emotional_urgency': {
                'priority': 'high',
                'route_to': 'senior_agent',
                'notification': 'immediate',
                'message': 'User shows high emotional distress - immediate attention required'
            },
            'repeated_issue': {
                'priority': 'high',
                'route_to': 'supervisor',
                'notification': 'immediate',
                'message': 'User reporting repeated/ongoing issues - escalation needed'
            },
            'threat_detected': {
                'priority': 'critical',
                'route_to': 'management',
                'notification': 'immediate',
                'message': 'Potential threat or legal concern detected - immediate management attention'
            },
            'sentiment_deterioration': {
                'priority': 'medium',
                'route_to': 'senior_agent',
                'notification': 'within_hour',
                'message': 'User sentiment deteriorating - preventive escalation recommended'
            }
        }
    
    def _create_chatbot_tables(self):
        """Create database tables for chatbot functionality"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chatbot_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id INTEGER,
                user_message TEXT,
                bot_response TEXT,
                detected_emotion TEXT,
                emotion_confidence REAL,
                sentiment_score REAL,
                escalation_triggered BOOLEAN DEFAULT FALSE,
                escalation_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Escalations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chatbot_escalations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id INTEGER,
                escalation_type TEXT,
                escalation_reason TEXT,
                priority_level TEXT,
                routed_to TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User emotional profiles
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_emotional_profiles (
                user_id INTEGER PRIMARY KEY,
                dominant_emotions TEXT,
                avg_sentiment REAL,
                escalation_count INTEGER DEFAULT 0,
                last_interaction TIMESTAMP,
                emotional_volatility REAL DEFAULT 0.0,
                satisfaction_trend REAL DEFAULT 0.0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_conversation_history(self):
        """Load recent conversation history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load recent conversations (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            cursor.execute('''
                SELECT session_id, user_id, user_message, detected_emotion, sentiment_score, created_at
                FROM chatbot_conversations 
                WHERE created_at >= ?
                ORDER BY created_at DESC
            ''', (cutoff_time,))
            
            conversations = cursor.fetchall()
            
            for conv in conversations:
                session_id, user_id, message, emotion, sentiment, timestamp = conv
                self.conversation_history[session_id].append({
                    'user_id': user_id,
                    'message': message,
                    'emotion': emotion,
                    'sentiment': sentiment,
                    'timestamp': timestamp
                })
            
            conn.close()
            logger.info(f"Loaded {len(conversations)} recent conversations")
            
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
    
    def analyze_emotion_and_intent(self, message: str) -> Dict[str, Any]:
        """Analyze emotion and intent from user message"""
        analysis = {
            'primary_emotion': 'neutral',
            'emotion_confidence': 0.0,
            'sentiment_score': 0.0,
            'intent': 'general_query',
            'urgency_level': 'normal',
            'escalation_indicators': [],
            'keywords': []
        }
        
        try:
            # Emotion detection using transformer model
            if self.emotion_classifier:
                emotion_result = self.emotion_classifier(message)
                if emotion_result:
                    analysis['primary_emotion'] = emotion_result[0]['label'].lower()
                    analysis['emotion_confidence'] = emotion_result[0]['score']
            
            # Sentiment analysis
            sentiment_scores = self.sentiment_analyzer.polarity_scores(message)
            analysis['sentiment_score'] = sentiment_scores['compound']
            
            # Intent detection
            analysis['intent'] = self._detect_intent(message)
            
            # Urgency detection
            analysis['urgency_level'] = self._detect_urgency(message)
            
            # Escalation indicators
            analysis['escalation_indicators'] = self._detect_escalation_indicators(message)
            
            # Extract keywords
            analysis['keywords'] = self._extract_keywords(message)
            
        except Exception as e:
            logger.error(f"Error in emotion/intent analysis: {e}")
        
        return analysis
    
    def _detect_intent(self, message: str) -> str:
        """Detect user intent from message"""
        message_lower = message.lower()
        
        # Intent patterns
        intent_patterns = {
            'check_status': ['status', 'track', 'update', 'progress', 'where is my', 'what happened to'],
            'submit_complaint': ['submit', 'new complaint', 'file grievance', 'report issue', 'complain about'],
            'request_escalation': ['escalate', 'supervisor', 'manager', 'speak to someone', 'higher up'],
            'express_dissatisfaction': ['unsatisfied', 'not happy', 'disappointed', 'poor service'],
            'request_timeline': ['when will', 'how long', 'timeline', 'deadline', 'eta'],
            'provide_feedback': ['feedback', 'suggestion', 'improve', 'better'],
            'request_information': ['how to', 'what is', 'explain', 'information about'],
            'emergency': ['emergency', 'urgent', 'critical', 'immediate help']
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                return intent
        
        return 'general_query'
    
    def _detect_urgency(self, message: str) -> str:
        """Detect urgency level from message"""
        message_lower = message.lower()
        
        # Urgency indicators
        critical_indicators = ['emergency', 'critical', 'life threatening', 'dangerous', 'unsafe']
        high_indicators = ['urgent', 'asap', 'immediate', 'right now', 'cannot wait']
        medium_indicators = ['soon', 'quickly', 'fast', 'priority', 'important']
        
        if any(indicator in message_lower for indicator in critical_indicators):
            return 'critical'
        elif any(indicator in message_lower for indicator in high_indicators):
            return 'high'
        elif any(indicator in message_lower for indicator in medium_indicators):
            return 'medium'
        else:
            return 'normal'
    
    def _detect_escalation_indicators(self, message: str) -> List[str]:
        """Detect escalation indicators in message"""
        message_lower = message.lower()
        indicators = []
        
        for category, phrases in self.escalation_phrases.items():
            if any(phrase in message_lower for phrase in phrases):
                indicators.append(category)
        
        return indicators
    
    def _extract_keywords(self, message: str) -> List[str]:
        """Extract important keywords from message"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', message.lower())
        
        # Filter out common words and keep meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'cannot', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'this', 'that', 'these', 'those'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        return keywords[:10]  # Return top 10 keywords
    
    def generate_response(self, message: str, session_id: str, user_id: int = None) -> Dict[str, Any]:
        """Generate empathetic response to user message"""
        # Analyze emotion and intent
        analysis = self.analyze_emotion_and_intent(message)
        
        # Update conversation history
        self.conversation_history[session_id].append({
            'user_id': user_id,
            'message': message,
            'emotion': analysis['primary_emotion'],
            'sentiment': analysis['sentiment_score'],
            'timestamp': datetime.now().isoformat()
        })
        
        # Check for escalation needs
        escalation_needed, escalation_reason = self._should_escalate(
            session_id, analysis, user_id
        )
        
        # Generate appropriate response
        response = self._generate_contextual_response(analysis, session_id, escalation_needed)
        
        # Log conversation
        self._log_conversation(
            session_id, user_id, message, response, analysis, 
            escalation_needed, escalation_reason
        )
        
        # Update user emotional profile
        if user_id:
            self._update_user_emotional_profile(user_id, analysis)
        
        result = {
            'response': response,
            'emotion_detected': analysis['primary_emotion'],
            'emotion_confidence': analysis['emotion_confidence'],
            'sentiment_score': analysis['sentiment_score'],
            'intent': analysis['intent'],
            'urgency_level': analysis['urgency_level'],
            'escalation_needed': escalation_needed,
            'escalation_reason': escalation_reason,
            'suggested_actions': self._get_suggested_actions(analysis, escalation_needed)
        }
        
        # Handle escalation if needed
        if escalation_needed:
            escalation_info = self._trigger_escalation(
                session_id, user_id, analysis, escalation_reason
            )
            result['escalation_info'] = escalation_info
        
        return result
    
    def _should_escalate(self, session_id: str, analysis: Dict, user_id: int = None) -> Tuple[bool, str]:
        """Determine if escalation is needed"""
        escalation_reasons = []
        
        # Check emotion intensity
        if (analysis['emotion_confidence'] > self.escalation_thresholds['emotion_intensity'] and
            analysis['primary_emotion'] in ['anger', 'frustration']):
            escalation_reasons.append('high_emotion_intensity')
        
        # Check for threat indicators
        if 'threat_indicators' in analysis['escalation_indicators']:
            escalation_reasons.append('threat_detected')
        
        # Check conversation history
        history = list(self.conversation_history[session_id])
        if len(history) >= self.escalation_thresholds['conversation_length']:
            # Check for repeated negative sentiment
            recent_sentiments = [h['sentiment'] for h in history[-3:]]
            if all(s < -0.2 for s in recent_sentiments):
                escalation_reasons.append('repeated_negative_sentiment')
        
        # Check for sentiment deterioration
        if len(history) >= 2:
            sentiment_change = analysis['sentiment_score'] - history[-2]['sentiment']
            if sentiment_change <= self.escalation_thresholds['sentiment_deterioration']:
                escalation_reasons.append('sentiment_deterioration')
        
        # Check urgency level
        if analysis['urgency_level'] in ['critical', 'high']:
            escalation_reasons.append('high_urgency')
        
        # Check for repeated frustration keywords
        frustration_count = len([i for i in analysis['escalation_indicators'] 
                               if i in ['high_intensity', 'repeated_frustration']])
        if frustration_count >= self.escalation_thresholds['frustration_keywords']:
            escalation_reasons.append('multiple_frustration_indicators')
        
        should_escalate = len(escalation_reasons) > 0
        escalation_reason = ', '.join(escalation_reasons) if escalation_reasons else None
        
        return should_escalate, escalation_reason
    
    def _generate_contextual_response(self, analysis: Dict, session_id: str, 
                                    escalation_needed: bool) -> str:
        """Generate contextually appropriate response"""
        emotion = analysis['primary_emotion']
        intent = analysis['intent']
        
        # Start with emotional acknowledgment
        response_parts = []
        
        # Emotional acknowledgment
        if emotion in self.emotional_responses:
            acknowledgment = random.choice(self.emotional_responses[emotion]['acknowledgment'])
            empathy = random.choice(self.emotional_responses[emotion]['empathy'])
            response_parts.extend([acknowledgment, empathy])
        
        # Intent-based response
        if intent in self.faq_responses:
            intent_response = random.choice(self.faq_responses[intent])
            response_parts.append(intent_response)
        elif intent == 'emergency':
            response_parts.append("This sounds like an urgent matter. I'm escalating this immediately to our emergency response team.")
        elif intent == 'express_dissatisfaction':
            response_parts.append("I sincerely apologize for the poor experience. Let me make sure this gets the attention it deserves.")
        else:
            # General helpful response
            response_parts.append("I'm here to help you resolve this matter. Could you provide me with more specific details about your concern?")
        
        # Action statement based on emotion
        if emotion in self.emotional_responses:
            action = random.choice(self.emotional_responses[emotion]['action'])
            response_parts.append(action)
        
        # Escalation notification
        if escalation_needed:
            escalation_msg = self._get_escalation_message(analysis)
            response_parts.append(escalation_msg)
        
        # Additional helpful information
        if intent == 'check_status':
            response_parts.append("If you have your grievance ID, I can provide you with detailed status information right away.")
        elif intent == 'submit_complaint':
            response_parts.append("I'll guide you through the process step by step to ensure your concern is properly documented.")
        
        return " ".join(response_parts)
    
    def _get_escalation_message(self, analysis: Dict) -> str:
        """Get appropriate escalation message"""
        urgency = analysis['urgency_level']
        emotion = analysis['primary_emotion']
        
        if urgency == 'critical':
            return "Given the critical nature of this issue, I'm immediately connecting you with our emergency response team."
        elif emotion in ['anger', 'frustration'] and analysis['emotion_confidence'] > 0.8:
            return "I understand this is very frustrating for you. I'm escalating this to our senior team who can provide you with immediate personalized attention."
        elif 'threat_indicators' in analysis.get('escalation_indicators', []):
            return "I'm connecting you with our management team to ensure this matter receives appropriate high-level attention."
        else:
            return "To ensure you receive the best possible service, I'm escalating this to a senior team member who can provide specialized assistance."
    
    def _get_suggested_actions(self, analysis: Dict, escalation_needed: bool) -> List[str]:
        """Get suggested actions based on analysis"""
        actions = []
        
        intent = analysis['intent']
        emotion = analysis['primary_emotion']
        urgency = analysis['urgency_level']
        
        if escalation_needed:
            actions.append("Escalate to appropriate team immediately")
        
        if urgency in ['critical', 'high']:
            actions.append("Prioritize for immediate response")
        
        if emotion in ['anger', 'frustration']:
            actions.append("Assign to experienced agent with empathy training")
        
        if intent == 'check_status':
            actions.append("Provide detailed status update with timeline")
        elif intent == 'submit_complaint':
            actions.append("Guide through complaint submission process")
        elif intent == 'request_escalation':
            actions.append("Connect with supervisor or senior agent")
        
        if 'repeated_frustration' in analysis.get('escalation_indicators', []):
            actions.append("Review case history for patterns")
        
        return actions
    
    def _trigger_escalation(self, session_id: str, user_id: int, 
                          analysis: Dict, escalation_reason: str) -> Dict[str, Any]:
        """Trigger escalation process"""
        escalation_type = self._determine_escalation_type(analysis, escalation_reason)
        escalation_action = self.escalation_actions.get(escalation_type, 
                                                      self.escalation_actions['emotional_urgency'])
        
        escalation_info = {
            'escalation_id': self._create_escalation_record(
                session_id, user_id, escalation_type, escalation_reason, escalation_action
            ),
            'priority': escalation_action['priority'],
            'routed_to': escalation_action['route_to'],
            'notification_urgency': escalation_action['notification'],
            'message': escalation_action['message'],
            'estimated_response_time': self._get_estimated_response_time(escalation_action['priority'])
        }
        
        # Update escalation count for user
        if user_id:
            self.escalation_triggers[user_id] += 1
        
        return escalation_info
    
    def _determine_escalation_type(self, analysis: Dict, escalation_reason: str) -> str:
        """Determine the type of escalation needed"""
        if 'threat_detected' in escalation_reason:
            return 'threat_detected'
        elif 'high_emotion_intensity' in escalation_reason:
            return 'emotional_urgency'
        elif 'repeated_negative_sentiment' in escalation_reason:
            return 'repeated_issue'
        elif 'sentiment_deterioration' in escalation_reason:
            return 'sentiment_deterioration'
        else:
            return 'emotional_urgency'  # Default
    
    def _create_escalation_record(self, session_id: str, user_id: int, 
                                escalation_type: str, escalation_reason: str,
                                escalation_action: Dict) -> int:
        """Create escalation record in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chatbot_escalations 
                (session_id, user_id, escalation_type, escalation_reason, priority_level, routed_to)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                user_id,
                escalation_type,
                escalation_reason,
                escalation_action['priority'],
                escalation_action['route_to']
            ))
            
            escalation_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return escalation_id
            
        except Exception as e:
            logger.error(f"Error creating escalation record: {e}")
            return 0
    
    def _get_estimated_response_time(self, priority: str) -> str:
        """Get estimated response time based on priority"""
        response_times = {
            'critical': '5-15 minutes',
            'high': '15-30 minutes',
            'medium': '30-60 minutes',
            'low': '1-2 hours'
        }
        return response_times.get(priority, '30-60 minutes')
    
    def _log_conversation(self, session_id: str, user_id: int, user_message: str,
                         bot_response: str, analysis: Dict, escalation_needed: bool,
                         escalation_reason: str):
        """Log conversation to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chatbot_conversations 
                (session_id, user_id, user_message, bot_response, detected_emotion, 
                 emotion_confidence, sentiment_score, escalation_triggered, escalation_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                user_id,
                user_message,
                bot_response,
                analysis['primary_emotion'],
                analysis['emotion_confidence'],
                analysis['sentiment_score'],
                escalation_needed,
                escalation_reason
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
    
    def _update_user_emotional_profile(self, user_id: int, analysis: Dict):
        """Update user's emotional profile"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get existing profile
            cursor.execute('SELECT * FROM user_emotional_profiles WHERE user_id = ?', (user_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing profile
                # Calculate new averages and update emotional volatility
                current_sentiment = existing[2] if existing[2] else 0
                new_avg_sentiment = (current_sentiment + analysis['sentiment_score']) / 2
                
                cursor.execute('''
                    UPDATE user_emotional_profiles 
                    SET avg_sentiment = ?, last_interaction = ?, emotional_volatility = ?
                    WHERE user_id = ?
                ''', (
                    new_avg_sentiment,
                    datetime.now(),
                    abs(analysis['sentiment_score'] - current_sentiment),
                    user_id
                ))
            else:
                # Create new profile
                cursor.execute('''
                    INSERT INTO user_emotional_profiles 
                    (user_id, dominant_emotions, avg_sentiment, last_interaction)
                    VALUES (?, ?, ?, ?)
                ''', (
                    user_id,
                    analysis['primary_emotion'],
                    analysis['sentiment_score'],
                    datetime.now()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating user emotional profile: {e}")
    
    def get_conversation_analytics(self, time_period_days: int = 7) -> Dict[str, Any]:
        """Get analytics on chatbot conversations"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            
            # Conversation volume
            conv_df = pd.read_sql_query('''
                SELECT detected_emotion, sentiment_score, escalation_triggered, 
                       date(created_at) as date, created_at
                FROM chatbot_conversations 
                WHERE created_at >= ?
            ''', conn, params=[cutoff_date])
            
            # Escalation data
            esc_df = pd.read_sql_query('''
                SELECT escalation_type, priority_level, status, created_at
                FROM chatbot_escalations 
                WHERE created_at >= ?
            ''', conn, params=[cutoff_date])
            
            conn.close()
            
            analytics = {
                'conversation_volume': {
                    'total_conversations': len(conv_df),
                    'daily_average': len(conv_df) / time_period_days,
                    'peak_day': conv_df.groupby('date').size().idxmax() if len(conv_df) > 0 else None
                },
                'emotion_distribution': conv_df['detected_emotion'].value_counts().to_dict() if len(conv_df) > 0 else {},
                'sentiment_trends': {
                    'average_sentiment': conv_df['sentiment_score'].mean() if len(conv_df) > 0 else 0,
                    'sentiment_range': [conv_df['sentiment_score'].min(), conv_df['sentiment_score'].max()] if len(conv_df) > 0 else [0, 0]
                },
                'escalation_metrics': {
                    'total_escalations': len(esc_df),
                    'escalation_rate': (esc_df['escalation_triggered'].sum() / len(conv_df) * 100) if len(conv_df) > 0 else 0,
                    'escalation_types': esc_df['escalation_type'].value_counts().to_dict() if len(esc_df) > 0 else {},
                    'priority_distribution': esc_df['priority_level'].value_counts().to_dict() if len(esc_df) > 0 else {}
                },
                'performance_metrics': {
                    'resolution_effectiveness': self._calculate_resolution_effectiveness(conv_df),
                    'user_satisfaction_proxy': self._calculate_satisfaction_proxy(conv_df)
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting conversation analytics: {e}")
            return {}
    
    def _calculate_resolution_effectiveness(self, df: pd.DataFrame) -> float:
        """Calculate chatbot resolution effectiveness"""
        if len(df) == 0:
            return 0.0
        
        # Simple metric: ratio of non-escalated to total conversations
        non_escalated = len(df[df['escalation_triggered'] == False])
        return (non_escalated / len(df)) * 100
    
    def _calculate_satisfaction_proxy(self, df: pd.DataFrame) -> float:
        """Calculate user satisfaction proxy based on sentiment improvement"""
        if len(df) == 0:
            return 0.0
        
        # Use average sentiment as proxy for satisfaction
        avg_sentiment = df['sentiment_score'].mean()
        
        # Normalize to 0-100 scale
        return max(0, min(100, (avg_sentiment + 1) * 50))
    
    def handle_feedback(self, session_id: str, feedback: str, rating: int) -> Dict[str, Any]:
        """Handle user feedback on chatbot interaction"""
        try:
            # Analyze feedback
            feedback_analysis = self.analyze_emotion_and_intent(feedback)
            
            # Store feedback (you might want to create a feedback table)
            # For now, we'll update the conversation record
            
            response = {
                'acknowledged': True,
                'message': "Thank you for your feedback. It helps us improve our service.",
                'rating_received': rating,
                'feedback_sentiment': feedback_analysis['sentiment_score']
            }
            
            if rating <= 2:
                response['message'] = "I'm sorry the interaction didn't meet your expectations. Your feedback will help us improve."
                response['follow_up_needed'] = True
            elif rating >= 4:
                response['message'] = "Thank you for the positive feedback! I'm glad I could help."
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling feedback: {e}")
            return {'acknowledged': False, 'error': 'Failed to process feedback'}
    
    def get_user_conversation_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get conversation history for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_message, bot_response, detected_emotion, sentiment_score, 
                       escalation_triggered, created_at
                FROM chatbot_conversations 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (user_id, limit))
            
            conversations = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'user_message': conv[0],
                    'bot_response': conv[1],
                    'emotion': conv[2],
                    'sentiment': conv[3],
                    'escalated': bool(conv[4]),
                    'timestamp': conv[5]
                }
                for conv in conversations
            ]
            
        except Exception as e:
            logger.error(f"Error getting user conversation history: {e}")
            return []
    
    def reset_conversation(self, session_id: str):
        """Reset conversation state for a session"""
        if session_id in self.conversation_history:
            self.conversation_history[session_id].clear()
        
        # Could also clear escalation triggers if needed
        
    def get_escalation_summary(self, time_period_days: int = 7) -> Dict[str, Any]:
        """Get summary of recent escalations"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            
            escalations_df = pd.read_sql_query('''
                SELECT escalation_type, escalation_reason, priority_level, 
                       routed_to, status, created_at
                FROM chatbot_escalations 
                WHERE created_at >= ?
                ORDER BY created_at DESC
            ''', conn, params=[cutoff_date])
            
            conn.close()
            
            if len(escalations_df) == 0:
                return {'total_escalations': 0, 'message': 'No escalations in the specified period'}
            
            summary = {
                'total_escalations': len(escalations_df),
                'escalation_rate_per_day': len(escalations_df) / time_period_days,
                'by_type': escalations_df['escalation_type'].value_counts().to_dict(),
                'by_priority': escalations_df['priority_level'].value_counts().to_dict(),
                'by_routing': escalations_df['routed_to'].value_counts().to_dict(),
                'by_status': escalations_df['status'].value_counts().to_dict(),
                'recent_escalations': escalations_df.head(5).to_dict('records')
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting escalation summary: {e}")
            return {'error': 'Failed to generate escalation summary'}
