"""
Advanced Emotion-Aware Sentiment Analyzer
Implements sophisticated emotion detection using BERT/RoBERTa models
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedEmotionSentimentAnalyzer:
    """
    Advanced sentiment analyzer using multiple models and approaches
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize various analyzers
        self._init_emotion_models()
        self._init_sentiment_models()
        self._init_keywords()
        self._init_ml_components()
        
        # Impact score weights
        self.impact_weights = {
            'emotion': 0.3,
            'length': 0.2,
            'keyword_severity': 0.25,
            'past_complaints': 0.15,
            'urgency_indicators': 0.1
        }
        
        # Learning history for adaptive ML
        self.resolution_history = []
        self.prediction_accuracy = 0.7  # Initial accuracy
        
    def _init_emotion_models(self):
        """Initialize emotion detection models"""
        try:
            # Emotion classification pipeline using pre-trained model
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Sarcasm detection model
            self.sarcasm_detector = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-irony",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Emotion models initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to load advanced emotion models: {e}")
            self.emotion_classifier = None
            self.sarcasm_detector = None
    
    def _init_sentiment_models(self):
        """Initialize sentiment analysis models"""
        try:
            # RoBERTa sentiment model
            self.roberta_sentiment = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # VADER for social media text
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            logger.info("Sentiment models initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to load sentiment models: {e}")
            self.roberta_sentiment = None
            self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def _init_keywords(self):
        """Initialize keyword dictionaries for various categories"""
        self.emotion_keywords = {
            'anger': ['angry', 'furious', 'rage', 'mad', 'pissed', 'livid', 'outraged', 'infuriated'],
            'sadness': ['sad', 'depressed', 'miserable', 'heartbroken', 'devastated', 'dejected'],
            'anxiety': ['anxious', 'worried', 'nervous', 'stressed', 'panic', 'overwhelmed'],
            'frustration': ['frustrated', 'annoyed', 'irritated', 'bothered', 'fed up'],
            'fear': ['afraid', 'scared', 'terrified', 'frightened', 'worried'],
            'disgust': ['disgusted', 'revolted', 'sickened', 'appalled']
        }
        
        self.urgency_keywords = [
            'urgent', 'immediate', 'emergency', 'critical', 'asap', 'now',
            'quickly', 'fast', 'soon', 'deadline', 'time-sensitive'
        ]
        
        self.severity_keywords = {
            'high': ['broken', 'failed', 'not working', 'dangerous', 'unsafe', 'serious'],
            'medium': ['issue', 'problem', 'concern', 'difficulty', 'trouble'],
            'low': ['suggestion', 'improvement', 'request', 'feedback']
        }
        
        self.sarcasm_indicators = [
            'great job', 'wonderful', 'perfect', 'amazing', 'brilliant',
            'fantastic', 'excellent', 'outstanding'
        ]
    
    def _init_ml_components(self):
        """Initialize machine learning components"""
        self.sentiment_history = {}  # User ID -> list of sentiments over time
        self.resolution_feedback = {}  # For learning from past resolutions
        
    def analyze_emotion_advanced(self, text: str) -> Dict[str, Any]:
        """
        Advanced emotion analysis using multiple approaches
        """
        emotions = {
            'primary_emotion': 'neutral',
            'secondary_emotions': [],
            'confidence': 0.0,
            'intensity': 0.0,
            'emotions_detected': {}
        }
        
        try:
            # Use transformer model for emotion detection
            if self.emotion_classifier:
                emotion_result = self.emotion_classifier(text)
                if emotion_result:
                    emotions['primary_emotion'] = emotion_result[0]['label'].lower()
                    emotions['confidence'] = emotion_result[0]['score']
                    
                    # Get all emotions above threshold
                    for result in emotion_result[:3]:  # Top 3 emotions
                        if result['score'] > 0.1:
                            emotions['emotions_detected'][result['label'].lower()] = result['score']
            
            # Keyword-based emotion detection as fallback/supplement
            keyword_emotions = self._detect_emotions_by_keywords(text)
            emotions['emotions_detected'].update(keyword_emotions)
            
            # Determine intensity based on text characteristics
            emotions['intensity'] = self._calculate_emotion_intensity(text)
            
            # Check for sarcasm
            emotions['sarcasm_detected'] = self._detect_sarcasm(text)
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            emotions = self._fallback_emotion_analysis(text)
        
        return emotions
    
    def _detect_emotions_by_keywords(self, text: str) -> Dict[str, float]:
        """Detect emotions using keyword matching"""
        text_lower = text.lower()
        detected_emotions = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                # Normalize by text length and keyword frequency
                score = min(count / len(keywords) * 2, 1.0)
                detected_emotions[emotion] = score
        
        return detected_emotions
    
    def _calculate_emotion_intensity(self, text: str) -> float:
        """Calculate emotional intensity based on various factors"""
        factors = []
        
        # Exclamation marks
        exclamation_count = text.count('!')
        factors.append(min(exclamation_count / 3, 1.0))
        
        # Capital letters
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        factors.append(min(caps_ratio * 3, 1.0))
        
        # Repetitive letters (e.g., "sooo bad")
        repetitive_pattern = len(re.findall(r'(.)\1{2,}', text.lower()))
        factors.append(min(repetitive_pattern / 5, 1.0))
        
        # Text length (longer complaints often indicate higher frustration)
        length_factor = min(len(text) / 500, 1.0)
        factors.append(length_factor)
        
        return np.mean(factors) if factors else 0.0
    
    def _detect_sarcasm(self, text: str) -> Dict[str, Any]:
        """Detect sarcasm in text"""
        sarcasm_info = {
            'detected': False,
            'confidence': 0.0,
            'indicators': []
        }
        
        try:
            # Use transformer model if available
            if self.sarcasm_detector:
                result = self.sarcasm_detector(text)
                if result and result[0]['label'] == 'IRONY':
                    sarcasm_info['detected'] = True
                    sarcasm_info['confidence'] = result[0]['score']
            
            # Keyword-based detection
            text_lower = text.lower()
            for indicator in self.sarcasm_indicators:
                if indicator in text_lower:
                    # Check context for negative sentiment
                    if any(neg in text_lower for neg in ['but', 'however', 'unfortunately', 'sadly']):
                        sarcasm_info['detected'] = True
                        sarcasm_info['indicators'].append(indicator)
        
        except Exception as e:
            logger.error(f"Error in sarcasm detection: {e}")
        
        return sarcasm_info
    
    def _fallback_emotion_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback emotion analysis using simpler methods"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity < -0.3:
            primary_emotion = 'anger' if 'angry' in text.lower() else 'sadness'
        elif polarity > 0.3:
            primary_emotion = 'joy'
        else:
            primary_emotion = 'neutral'
        
        return {
            'primary_emotion': primary_emotion,
            'confidence': abs(polarity),
            'intensity': abs(polarity),
            'emotions_detected': {primary_emotion: abs(polarity)},
            'sarcasm_detected': {'detected': False, 'confidence': 0.0}
        }
    
    def calculate_impact_score(self, text: str, emotion_data: Dict, user_history: Dict = None) -> float:
        """
        Calculate comprehensive impact score for grievance prioritization
        """
        scores = {}
        
        # Emotion impact (0.3 weight)
        emotion_score = self._calculate_emotion_impact(emotion_data)
        scores['emotion'] = emotion_score
        
        # Text length impact (0.2 weight)
        length_score = min(len(text) / 1000, 1.0)  # Normalize to max 1000 chars
        scores['length'] = length_score
        
        # Keyword severity impact (0.25 weight)
        severity_score = self._calculate_keyword_severity(text)
        scores['keyword_severity'] = severity_score
        
        # Past complaints impact (0.15 weight)
        past_score = self._calculate_past_complaints_impact(user_history)
        scores['past_complaints'] = past_score
        
        # Urgency indicators impact (0.1 weight)
        urgency_score = self._calculate_urgency_score(text)
        scores['urgency_indicators'] = urgency_score
        
        # Calculate weighted final score
        impact_score = sum(
            scores[factor] * weight 
            for factor, weight in self.impact_weights.items()
        )
        
        return round(min(max(impact_score, 0.0), 1.0), 3)
    
    def _calculate_emotion_impact(self, emotion_data: Dict) -> float:
        """Calculate impact score based on emotions detected"""
        emotion_weights = {
            'anger': 1.0,
            'frustration': 0.9,
            'anxiety': 0.8,
            'sadness': 0.7,
            'fear': 0.8,
            'disgust': 0.85,
            'joy': 0.1,
            'neutral': 0.3
        }
        
        if not emotion_data.get('emotions_detected'):
            return 0.3
        
        max_impact = 0.0
        for emotion, confidence in emotion_data['emotions_detected'].items():
            weight = emotion_weights.get(emotion, 0.5)
            impact = weight * confidence * emotion_data.get('intensity', 0.5)
            max_impact = max(max_impact, impact)
        
        # Boost for sarcasm (often indicates hidden frustration)
        if emotion_data.get('sarcasm_detected', {}).get('detected', False):
            max_impact = min(max_impact * 1.3, 1.0)
        
        return max_impact
    
    def _calculate_keyword_severity(self, text: str) -> float:
        """Calculate severity based on keywords present"""
        text_lower = text.lower()
        severity_score = 0.0
        
        # High severity keywords
        high_count = sum(1 for keyword in self.severity_keywords['high'] if keyword in text_lower)
        severity_score += high_count * 0.3
        
        # Medium severity keywords  
        medium_count = sum(1 for keyword in self.severity_keywords['medium'] if keyword in text_lower)
        severity_score += medium_count * 0.2
        
        # Low severity keywords (reduce score)
        low_count = sum(1 for keyword in self.severity_keywords['low'] if keyword in text_lower)
        severity_score -= low_count * 0.1
        
        return min(max(severity_score, 0.0), 1.0)
    
    def _calculate_past_complaints_impact(self, user_history: Dict = None) -> float:
        """Calculate impact based on user's complaint history"""
        if not user_history:
            return 0.0
        
        unresolved_count = user_history.get('unresolved_complaints', 0)
        total_complaints = user_history.get('total_complaints', 1)
        avg_resolution_time = user_history.get('avg_resolution_time', 48)  # hours
        
        # Higher impact for users with many unresolved complaints
        unresolved_impact = min(unresolved_count / 5, 1.0)
        
        # Higher impact for users with slow resolution history
        time_impact = min(avg_resolution_time / 168, 1.0)  # Normalize by 1 week
        
        # Frequent complainers get moderate boost
        frequency_impact = min(total_complaints / 10, 0.5)
        
        return (unresolved_impact * 0.5 + time_impact * 0.3 + frequency_impact * 0.2)
    
    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency based on time-sensitive keywords"""
        text_lower = text.lower()
        urgency_count = sum(1 for keyword in self.urgency_keywords if keyword in text_lower)
        
        # Check for specific time mentions
        time_patterns = [
            r'\b(today|tonight|now|immediately)\b',
            r'\b(urgent|emergency|critical)\b',
            r'\b(deadline|due|expires?)\b',
            r'\b(asap|a\.s\.a\.p\.)\b'
        ]
        
        time_mentions = sum(1 for pattern in time_patterns if re.search(pattern, text_lower))
        
        total_urgency = urgency_count + time_mentions
        return min(total_urgency / 3, 1.0)
    
    def track_sentiment_shift(self, user_id: str, current_sentiment: str, 
                             grievance_id: str = None) -> Dict[str, Any]:
        """
        Track sentiment changes over time for escalation
        """
        if user_id not in self.sentiment_history:
            self.sentiment_history[user_id] = []
        
        # Add current sentiment with timestamp
        sentiment_entry = {
            'sentiment': current_sentiment,
            'timestamp': datetime.now(),
            'grievance_id': grievance_id
        }
        
        self.sentiment_history[user_id].append(sentiment_entry)
        
        # Keep only last 10 entries
        self.sentiment_history[user_id] = self.sentiment_history[user_id][-10:]
        
        # Analyze shift patterns
        shift_analysis = self._analyze_sentiment_progression(user_id)
        
        return shift_analysis
    
    def _analyze_sentiment_progression(self, user_id: str) -> Dict[str, Any]:
        """Analyze sentiment progression for a user"""
        history = self.sentiment_history.get(user_id, [])
        
        if len(history) < 2:
            return {
                'shift_detected': False,
                'escalation_needed': False,
                'trend': 'insufficient_data'
            }
        
        # Convert sentiments to numeric values for trend analysis
        sentiment_values = {
            'positive': 1,
            'neutral': 0, 
            'negative': -1
        }
        
        values = [sentiment_values.get(entry['sentiment'], 0) for entry in history]
        
        # Calculate trend
        if len(values) >= 3:
            recent_trend = np.mean(values[-3:]) - np.mean(values[-6:-3]) if len(values) >= 6 else np.mean(values[-3:])
        else:
            recent_trend = values[-1] - values[0]
        
        # Detect specific shift patterns
        escalation_patterns = [
            # Neutral/positive to negative
            (values[-2] >= 0 and values[-1] < 0),
            # Consistent decline over 3+ entries
            (len(values) >= 3 and all(values[i] >= values[i+1] for i in range(-3, -1))),
            # Multiple negative in a row after positive/neutral
            (len(values) >= 4 and values[-4] >= 0 and all(v < 0 for v in values[-3:]))
        ]
        
        shift_detected = any(escalation_patterns)
        escalation_needed = shift_detected and recent_trend < -0.5
        
        return {
            'shift_detected': shift_detected,
            'escalation_needed': escalation_needed,
            'trend': 'declining' if recent_trend < -0.2 else 'improving' if recent_trend > 0.2 else 'stable',
            'recent_sentiment_avg': np.mean(values[-3:]) if len(values) >= 3 else values[-1],
            'recommendation': self._get_escalation_recommendation(shift_detected, escalation_needed)
        }
    
    def _get_escalation_recommendation(self, shift_detected: bool, escalation_needed: bool) -> str:
        """Get recommendation based on sentiment analysis"""
        if escalation_needed:
            return "IMMEDIATE_ESCALATION: User sentiment severely deteriorated. Escalate to senior management."
        elif shift_detected:
            return "PRIORITY_REVIEW: User sentiment declining. Prioritize for faster resolution."
        else:
            return "NORMAL_PROCESSING: No concerning sentiment patterns detected."
    
    def learn_from_resolution(self, grievance_data: Dict, resolution_data: Dict):
        """
        Learn from past resolutions to improve future predictions
        """
        learning_entry = {
            'predicted_priority': grievance_data.get('predicted_priority'),
            'actual_priority': resolution_data.get('actual_priority'),
            'predicted_impact': grievance_data.get('impact_score'),
            'resolution_time': resolution_data.get('resolution_time_hours'),
            'user_satisfaction': resolution_data.get('satisfaction_rating'),
            'admin_feedback': resolution_data.get('admin_feedback'),
            'timestamp': datetime.now()
        }
        
        self.resolution_history.append(learning_entry)
        
        # Update prediction accuracy
        if len(self.resolution_history) >= 10:
            recent_predictions = self.resolution_history[-10:]
            correct_predictions = sum(
                1 for entry in recent_predictions 
                if entry['predicted_priority'] == entry['actual_priority']
            )
            self.prediction_accuracy = correct_predictions / len(recent_predictions)
        
        # Adjust weights based on learning
        self._adjust_impact_weights()
    
    def _adjust_impact_weights(self):
        """Adjust impact score weights based on learning history"""
        if len(self.resolution_history) < 20:
            return  # Need sufficient data
        
        # Analyze which factors correlate best with user satisfaction
        recent_data = self.resolution_history[-20:]
        
        # This is a simplified adjustment - in practice, you'd use more sophisticated ML
        high_satisfaction = [entry for entry in recent_data if entry.get('user_satisfaction', 0) >= 4]
        
        if len(high_satisfaction) > 5:
            # Increase weight of factors that led to high satisfaction
            # This is a placeholder for more complex learning logic
            pass
    
    def get_comprehensive_analysis(self, text: str, user_id: str = None, 
                                 user_history: Dict = None) -> Dict[str, Any]:
        """
        Get comprehensive analysis combining all features
        """
        # Basic sentiment analysis
        sentiment_basic = self._analyze_basic_sentiment(text)
        
        # Advanced emotion analysis
        emotion_analysis = self.analyze_emotion_advanced(text)
        
        # Calculate impact score
        impact_score = self.calculate_impact_score(text, emotion_analysis, user_history)
        
        # Sentiment shift tracking (if user_id provided)
        sentiment_shift = None
        if user_id:
            sentiment_shift = self.track_sentiment_shift(user_id, sentiment_basic['sentiment'])
        
        # Determine final priority
        priority = self._determine_smart_priority(
            sentiment_basic, emotion_analysis, impact_score, sentiment_shift
        )
        
        return {
            'basic_sentiment': sentiment_basic,
            'emotion_analysis': emotion_analysis,
            'impact_score': impact_score,
            'priority': priority,
            'sentiment_shift': sentiment_shift,
            'recommendation': self._get_handling_recommendation(emotion_analysis, impact_score, priority),
            'analysis_timestamp': datetime.now().isoformat(),
            'confidence': self._calculate_overall_confidence(sentiment_basic, emotion_analysis)
        }
    
    def _analyze_basic_sentiment(self, text: str) -> Dict[str, Any]:
        """Basic sentiment analysis using multiple models"""
        results = {}
        
        # TextBlob
        blob = TextBlob(text)
        results['textblob'] = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
        # VADER
        vader_scores = self.vader_analyzer.polarity_scores(text)
        results['vader'] = vader_scores
        
        # RoBERTa (if available)
        if self.roberta_sentiment:
            try:
                roberta_result = self.roberta_sentiment(text)
                results['roberta'] = roberta_result[0] if roberta_result else None
            except Exception as e:
                logger.error(f"RoBERTa sentiment error: {e}")
                results['roberta'] = None
        
        # Determine consensus sentiment
        sentiment = self._determine_consensus_sentiment(results)
        
        return {
            'sentiment': sentiment,
            'models': results,
            'confidence': self._calculate_sentiment_confidence(results)
        }
    
    def _determine_consensus_sentiment(self, results: Dict) -> str:
        """Determine consensus sentiment from multiple models"""
        scores = []
        
        # TextBlob
        if 'textblob' in results:
            scores.append(results['textblob']['polarity'])
        
        # VADER
        if 'vader' in results:
            scores.append(results['vader']['compound'])
        
        # RoBERTa
        if results.get('roberta'):
            label = results['roberta']['label']
            score = results['roberta']['score']
            if label == 'NEGATIVE':
                scores.append(-score)
            elif label == 'POSITIVE':
                scores.append(score)
            else:
                scores.append(0)
        
        if not scores:
            return 'neutral'
        
        avg_score = np.mean(scores)
        
        if avg_score < -0.1:
            return 'negative'
        elif avg_score > 0.1:
            return 'positive'
        else:
            return 'neutral'
    
    def _calculate_sentiment_confidence(self, results: Dict) -> float:
        """Calculate confidence in sentiment analysis"""
        if not results:
            return 0.0
        
        scores = []
        for model, data in results.items():
            if model == 'textblob':
                scores.append(abs(data['polarity']))
            elif model == 'vader':
                scores.append(abs(data['compound']))
            elif model == 'roberta' and data:
                scores.append(data['score'])
        
        return np.mean(scores) if scores else 0.0
    
    def _determine_smart_priority(self, sentiment: Dict, emotion: Dict, 
                                impact_score: float, sentiment_shift: Dict = None) -> int:
        """Determine priority using comprehensive analysis"""
        base_priority = 3  # Default low priority
        
        # Adjust based on sentiment
        if sentiment['sentiment'] == 'negative':
            base_priority = 2
        elif sentiment['sentiment'] == 'positive':
            base_priority = 3
        
        # Adjust based on emotions
        high_priority_emotions = ['anger', 'frustration', 'anxiety', 'fear']
        if any(emotion in emotion.get('emotions_detected', {}) for emotion in high_priority_emotions):
            base_priority = min(base_priority, 1)
        
        # Adjust based on impact score
        if impact_score >= 0.8:
            base_priority = 1
        elif impact_score >= 0.6:
            base_priority = min(base_priority, 2)
        
        # Adjust based on sentiment shift
        if sentiment_shift and sentiment_shift.get('escalation_needed'):
            base_priority = 1
        
        return max(1, min(base_priority, 3))
    
    def _get_handling_recommendation(self, emotion: Dict, impact_score: float, priority: int) -> str:
        """Get recommendation for handling the grievance"""
        recommendations = []
        
        if priority == 1:
            recommendations.append("HIGH PRIORITY - Immediate attention required")
        
        if emotion.get('sarcasm_detected', {}).get('detected'):
            recommendations.append("Sarcasm detected - Handle with extra care and empathy")
        
        primary_emotion = emotion.get('primary_emotion', 'neutral')
        if primary_emotion == 'anger':
            recommendations.append("User is angry - Ensure immediate acknowledgment and swift action")
        elif primary_emotion == 'anxiety':
            recommendations.append("User is anxious - Provide clear timeline and frequent updates")
        elif primary_emotion == 'sadness':
            recommendations.append("User is sad - Show empathy and provide comprehensive support")
        
        if impact_score >= 0.8:
            recommendations.append("High impact score - May affect multiple users or systems")
        
        return " | ".join(recommendations) if recommendations else "Standard processing recommended"
    
    def _calculate_overall_confidence(self, sentiment: Dict, emotion: Dict) -> float:
        """Calculate overall confidence in the analysis"""
        sentiment_conf = sentiment.get('confidence', 0.0)
        emotion_conf = emotion.get('confidence', 0.0)
        
        return (sentiment_conf + emotion_conf) / 2

    def save_model_state(self, filepath: str):
        """Save the current state of the analyzer"""
        state = {
            'sentiment_history': self.sentiment_history,
            'resolution_history': self.resolution_history,
            'prediction_accuracy': self.prediction_accuracy,
            'impact_weights': self.impact_weights
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_model_state(self, filepath: str):
        """Load previously saved state"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            self.sentiment_history = state.get('sentiment_history', {})
            self.resolution_history = state.get('resolution_history', [])
            self.prediction_accuracy = state.get('prediction_accuracy', 0.7)
            self.impact_weights = state.get('impact_weights', self.impact_weights)
