"""
Resolution Quality Predictor
Evaluates response effectiveness before sending to users
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import re
import logging
from textblob import TextBlob
import pickle
import os
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)

class ResolutionQualityPredictor:
    """
    AI-powered system to predict and improve response quality
    """
    
    def __init__(self, db_path: str = 'grievance_system.db'):
        self.db_path = db_path
        
        # Initialize models
        self.quality_model = None
        self.satisfaction_model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Quality assessment criteria
        self.quality_criteria = {
            'tone_appropriateness': 0.25,
            'completeness': 0.25,
            'clarity': 0.20,
            'empathy': 0.15,
            'actionability': 0.15
        }
        
        # Response templates and patterns
        self._init_response_patterns()
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            self.sentiment_pipeline = None
        
        # Load historical data and train models
        self._load_and_train_models()
    
    def _init_response_patterns(self):
        """Initialize response quality patterns and templates"""
        
        # High-quality response indicators
        self.quality_indicators = {
            'acknowledgment': [
                'thank you for', 'we appreciate', 'i understand', 'we recognize',
                'thank you for bringing this to our attention', 'i acknowledge'
            ],
            'empathy': [
                'i understand how', 'this must be', 'i can imagine', 'i realize',
                'i appreciate your frustration', 'i empathize', 'i sympathize'
            ],
            'action_words': [
                'will investigate', 'will look into', 'will follow up', 'will contact',
                'will resolve', 'will address', 'taking action', 'working on'
            ],
            'specificity': [
                'within', 'by', 'on', 'specific', 'detailed', 'particular',
                'exactly', 'precisely', 'timeline', 'deadline'
            ],
            'professional_language': [
                'sincerely', 'regards', 'respectfully', 'please', 'kindly',
                'we value', 'we strive', 'our commitment'
            ]
        }
        
        # Low-quality response indicators
        self.quality_detractors = {
            'dismissive': [
                'not our fault', 'nothing we can do', 'policy states', 'unable to help',
                'that\'s not possible', 'we don\'t handle'
            ],
            'generic': [
                'thank you for contacting us', 'we will get back to you',
                'standard procedure', 'as per policy', 'unfortunately'
            ],
            'unclear': [
                'maybe', 'possibly', 'might', 'could be', 'not sure',
                'i think', 'perhaps', 'it depends'
            ],
            'unprofessional': [
                'asap', 'FYI', 'btw', 'lol', 'ok', 'yeah', 'nope'
            ]
        }
        
        # Tone analysis patterns
        self.tone_patterns = {
            'professional': [
                'dear', 'sincerely', 'respectfully', 'please', 'kindly',
                'we appreciate', 'thank you', 'regards'
            ],
            'empathetic': [
                'understand', 'sorry', 'apologize', 'concern', 'frustration',
                'disappointed', 'inconvenience', 'regret'
            ],
            'solution_oriented': [
                'resolve', 'fix', 'solution', 'address', 'improve',
                'correct', 'rectify', 'investigate'
            ]
        }
    
    def _load_and_train_models(self):
        """Load historical data and train prediction models"""
        try:
            historical_data = self._get_historical_response_data()
            
            if len(historical_data) < 10:
                logger.warning("Insufficient historical data for model training")
                return
            
            # Prepare features and targets
            X, y_quality, y_satisfaction = self._prepare_training_data(historical_data)
            
            if len(X) > 0:
                # Train quality prediction model
                self.quality_model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.quality_model.fit(X, y_quality)
                
                # Train satisfaction prediction model
                self.satisfaction_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                self.satisfaction_model.fit(X, y_satisfaction)
                
                logger.info("Quality prediction models trained successfully")
                
                # Save models
                self._save_models()
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def _get_historical_response_data(self) -> pd.DataFrame:
        """Get historical response data with ratings"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get grievances with responses and ratings
            query = '''
                SELECT g.id, g.description as complaint, g.response, g.rating, 
                       g.feedback, g.category, g.priority, g.sentiment,
                       g.created_at, g.resolved_at
                FROM grievances g
                WHERE g.response IS NOT NULL 
                AND g.rating IS NOT NULL
                AND g.response != ''
                ORDER BY g.resolved_at DESC
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data with features and targets"""
        if len(df) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Extract features for each response
        features = []
        quality_scores = []
        satisfaction_scores = []
        
        for _, row in df.iterrows():
            # Extract response features
            response_features = self._extract_response_features(
                row['response'], row['complaint'], row['category'], 
                row['priority'], row['sentiment']
            )
            
            # Calculate quality score from rating and feedback
            quality_score = self._calculate_quality_score_from_rating(
                row['rating'], row['feedback']
            )
            
            features.append(response_features)
            quality_scores.append(quality_score)
            satisfaction_scores.append(row['rating'] / 5.0)  # Normalize to 0-1
        
        return np.array(features), np.array(quality_scores), np.array(satisfaction_scores)
    
    def _extract_response_features(self, response: str, complaint: str, 
                                 category: str, priority: int, sentiment: str) -> List[float]:
        """Extract numerical features from response for ML model"""
        features = []
        
        # Basic text features
        features.extend([
            len(response),  # Response length
            len(response.split()),  # Word count
            len(response.split('.')) - 1,  # Sentence count
            np.mean([len(word) for word in response.split()]) if response.split() else 0,  # Avg word length
        ])
        
        # Quality indicator counts
        for category_name, indicators in self.quality_indicators.items():
            count = sum(1 for indicator in indicators if indicator.lower() in response.lower())
            features.append(count)
        
        # Quality detractor counts
        for category_name, detractors in self.quality_detractors.items():
            count = sum(1 for detractor in detractors if detractor.lower() in response.lower())
            features.append(-count)  # Negative because they detract from quality
        
        # Tone analysis
        for tone_name, patterns in self.tone_patterns.items():
            count = sum(1 for pattern in patterns if pattern.lower() in response.lower())
            features.append(count)
        
        # Sentiment analysis
        if self.sentiment_pipeline:
            try:
                sent_result = self.sentiment_pipeline(response[:512])  # Limit text length
                sentiment_score = sent_result[0]['score'] if sent_result[0]['label'] == 'POSITIVE' else -sent_result[0]['score']
                features.append(sentiment_score)
            except:
                features.append(0.0)
        else:
            blob = TextBlob(response)
            features.append(blob.sentiment.polarity)
        
        # Context features
        features.extend([
            priority,  # Priority level
            1 if sentiment == 'negative' else 0 if sentiment == 'neutral' else -1,  # Sentiment encoding
            len(complaint.split()),  # Complaint length
        ])
        
        # Category encoding (simplified)
        category_mapping = {'academic': 1, 'hostel': 2, 'infrastructure': 3, 'administration': 4, 'other': 5}
        features.append(category_mapping.get(category.lower(), 5))
        
        return features
    
    def _calculate_quality_score_from_rating(self, rating: float, feedback: str) -> float:
        """Calculate quality score from user rating and feedback"""
        # Base score from rating
        base_score = rating / 5.0
        
        # Adjust based on feedback sentiment if available
        if feedback and len(feedback.strip()) > 10:
            blob = TextBlob(feedback)
            feedback_sentiment = blob.sentiment.polarity
            
            # Positive feedback boosts score, negative feedback lowers it
            sentiment_adjustment = feedback_sentiment * 0.1
            base_score = min(max(base_score + sentiment_adjustment, 0.0), 1.0)
        
        return base_score
    
    def predict_response_quality(self, response: str, complaint: str, 
                               category: str = 'other', priority: int = 2, 
                               sentiment: str = 'neutral') -> Dict[str, Any]:
        """Predict the quality and effectiveness of a response"""
        try:
            # Comprehensive quality analysis
            analysis_result = {
                'overall_score': 0.0,
                'predicted_satisfaction': 0.0,
                'component_scores': {},
                'recommendations': [],
                'quality_level': 'poor',
                'confidence': 0.0,
                'detailed_analysis': {}
            }
            
            # Rule-based quality analysis
            component_scores = self._analyze_response_components(response, complaint)
            analysis_result['component_scores'] = component_scores
            
            # Calculate overall score from components
            overall_score = sum(
                score * weight for score, weight in 
                zip(component_scores.values(), self.quality_criteria.values())
            )
            analysis_result['overall_score'] = overall_score
            
            # ML-based prediction if models are available
            if self.quality_model and self.satisfaction_model:
                features = self._extract_response_features(
                    response, complaint, category, priority, sentiment
                )
                
                ml_quality = self.quality_model.predict([features])[0]
                ml_satisfaction = self.satisfaction_model.predict([features])[0]
                
                # Combine rule-based and ML predictions
                analysis_result['overall_score'] = (overall_score + ml_quality) / 2
                analysis_result['predicted_satisfaction'] = ml_satisfaction
                analysis_result['confidence'] = min(0.8, max(0.3, abs(overall_score - ml_quality)))
            else:
                analysis_result['predicted_satisfaction'] = overall_score
                analysis_result['confidence'] = 0.6  # Lower confidence without ML
            
            # Determine quality level
            if analysis_result['overall_score'] >= 0.8:
                analysis_result['quality_level'] = 'excellent'
            elif analysis_result['overall_score'] >= 0.6:
                analysis_result['quality_level'] = 'good'
            elif analysis_result['overall_score'] >= 0.4:
                analysis_result['quality_level'] = 'fair'
            else:
                analysis_result['quality_level'] = 'poor'
            
            # Generate recommendations
            analysis_result['recommendations'] = self._generate_improvement_recommendations(
                response, component_scores, complaint
            )
            
            # Detailed analysis
            analysis_result['detailed_analysis'] = self._detailed_response_analysis(response)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error predicting response quality: {e}")
            return {
                'overall_score': 0.5,
                'predicted_satisfaction': 0.5,
                'component_scores': {},
                'recommendations': ['Error in analysis - please review manually'],
                'quality_level': 'unknown',
                'confidence': 0.0
            }
    
    def _analyze_response_components(self, response: str, complaint: str) -> Dict[str, float]:
        """Analyze individual components of response quality"""
        components = {}
        
        # Tone appropriateness
        components['tone_appropriateness'] = self._assess_tone_appropriateness(response)
        
        # Completeness
        components['completeness'] = self._assess_completeness(response, complaint)
        
        # Clarity
        components['clarity'] = self._assess_clarity(response)
        
        # Empathy
        components['empathy'] = self._assess_empathy(response)
        
        # Actionability
        components['actionability'] = self._assess_actionability(response)
        
        return components
    
    def _assess_tone_appropriateness(self, response: str) -> float:
        """Assess if the tone is appropriate and professional"""
        score = 0.5  # Base score
        
        # Check for professional language
        professional_count = sum(
            1 for phrase in self.tone_patterns['professional']
            if phrase.lower() in response.lower()
        )
        score += min(professional_count * 0.1, 0.3)
        
        # Check for empathetic language
        empathy_count = sum(
            1 for phrase in self.tone_patterns['empathetic']
            if phrase.lower() in response.lower()
        )
        score += min(empathy_count * 0.1, 0.2)
        
        # Penalize unprofessional language
        unprofessional_count = sum(
            1 for phrase in self.quality_detractors['unprofessional']
            if phrase.lower() in response.lower()
        )
        score -= min(unprofessional_count * 0.2, 0.4)
        
        # Check sentiment
        blob = TextBlob(response)
        if blob.sentiment.polarity < -0.3:  # Very negative tone
            score -= 0.2
        elif blob.sentiment.polarity > 0.1:  # Positive tone
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _assess_completeness(self, response: str, complaint: str) -> float:
        """Assess if the response addresses the complaint completely"""
        score = 0.5
        
        # Extract key topics from complaint
        complaint_words = set(complaint.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'cannot'}
        complaint_words -= stop_words
        response_words -= stop_words
        
        # Calculate overlap
        if complaint_words:
            overlap_ratio = len(complaint_words.intersection(response_words)) / len(complaint_words)
            score += overlap_ratio * 0.3
        
        # Check for acknowledgment
        acknowledgment_count = sum(
            1 for phrase in self.quality_indicators['acknowledgment']
            if phrase.lower() in response.lower()
        )
        if acknowledgment_count > 0:
            score += 0.2
        
        # Check for specific action items
        action_count = sum(
            1 for phrase in self.quality_indicators['action_words']
            if phrase.lower() in response.lower()
        )
        score += min(action_count * 0.1, 0.2)
        
        # Minimum length check
        if len(response.split()) < 20:
            score -= 0.3
        elif len(response.split()) < 10:
            score -= 0.5
        
        return min(max(score, 0.0), 1.0)
    
    def _assess_clarity(self, response: str) -> float:
        """Assess clarity and readability of the response"""
        score = 0.5
        
        # Sentence structure
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        if sentences:
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            
            # Optimal sentence length is 15-25 words
            if 15 <= avg_sentence_length <= 25:
                score += 0.2
            elif 10 <= avg_sentence_length <= 30:
                score += 0.1
            elif avg_sentence_length > 40:
                score -= 0.2
        
        # Check for unclear language
        unclear_count = sum(
            1 for phrase in self.quality_detractors['unclear']
            if phrase.lower() in response.lower()
        )
        score -= min(unclear_count * 0.15, 0.3)
        
        # Check for specific details
        specificity_count = sum(
            1 for phrase in self.quality_indicators['specificity']
            if phrase.lower() in response.lower()
        )
        score += min(specificity_count * 0.1, 0.2)
        
        # Check for bullet points or numbered lists (good for clarity)
        if re.search(r'(\n\s*[-•*]\s+|\n\s*\d+\.\s+)', response):
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _assess_empathy(self, response: str) -> float:
        """Assess empathy and emotional intelligence in response"""
        score = 0.5
        
        # Check for empathetic language
        empathy_count = sum(
            1 for phrase in self.quality_indicators['empathy']
            if phrase.lower() in response.lower()
        )
        score += min(empathy_count * 0.2, 0.4)
        
        # Check for dismissive language
        dismissive_count = sum(
            1 for phrase in self.quality_detractors['dismissive']
            if phrase.lower() in response.lower()
        )
        score -= min(dismissive_count * 0.3, 0.6)
        
        # Check for emotional words
        emotional_words = ['understand', 'sorry', 'apologize', 'regret', 'concern', 'care', 'value']
        emotional_count = sum(
            1 for word in emotional_words
            if word in response.lower()
        )
        score += min(emotional_count * 0.05, 0.2)
        
        return min(max(score, 0.0), 1.0)
    
    def _assess_actionability(self, response: str) -> float:
        """Assess if the response provides clear next steps"""
        score = 0.5
        
        # Check for action words
        action_count = sum(
            1 for phrase in self.quality_indicators['action_words']
            if phrase.lower() in response.lower()
        )
        score += min(action_count * 0.15, 0.3)
        
        # Check for timelines
        timeline_patterns = [
            r'\b(within|by|on|before)\s+\d+\s+(days?|hours?|weeks?|months?)',
            r'\b(today|tomorrow|next week|this week|monday|tuesday|wednesday|thursday|friday)',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        ]
        
        for pattern in timeline_patterns:
            if re.search(pattern, response.lower()):
                score += 0.2
                break
        
        # Check for contact information or next steps
        contact_patterns = [
            r'\b(contact|call|email|reach out)\b',
            r'\b(follow up|follow-up|update)\b',
            r'\b(next step|next steps)\b'
        ]
        
        for pattern in contact_patterns:
            if re.search(pattern, response.lower()):
                score += 0.1
                break
        
        # Penalize generic responses
        generic_count = sum(
            1 for phrase in self.quality_detractors['generic']
            if phrase.lower() in response.lower()
        )
        score -= min(generic_count * 0.1, 0.2)
        
        return min(max(score, 0.0), 1.0)
    
    def _generate_improvement_recommendations(self, response: str, 
                                           component_scores: Dict[str, float],
                                           complaint: str) -> List[str]:
        """Generate specific recommendations for improving the response"""
        recommendations = []
        
        # Tone recommendations
        if component_scores.get('tone_appropriateness', 0) < 0.6:
            recommendations.append("Consider using more professional and empathetic language")
        
        # Completeness recommendations
        if component_scores.get('completeness', 0) < 0.6:
            recommendations.append("Address more aspects of the original complaint")
            recommendations.append("Acknowledge specific points raised by the user")
        
        # Clarity recommendations
        if component_scores.get('clarity', 0) < 0.6:
            recommendations.append("Break down complex information into clearer, shorter sentences")
            recommendations.append("Use bullet points or numbered lists for multiple action items")
        
        # Empathy recommendations
        if component_scores.get('empathy', 0) < 0.6:
            recommendations.append("Show more understanding of the user's frustration or concern")
            recommendations.append("Avoid language that might sound dismissive or bureaucratic")
        
        # Actionability recommendations
        if component_scores.get('actionability', 0) < 0.6:
            recommendations.append("Provide clearer next steps or action items")
            recommendations.append("Include specific timelines for resolution")
            recommendations.append("Offer contact information for follow-up")
        
        # Length-based recommendations
        word_count = len(response.split())
        if word_count < 20:
            recommendations.append("Provide a more detailed response to fully address the concern")
        elif word_count > 200:
            recommendations.append("Consider making the response more concise while retaining key information")
        
        # Generic response detection
        if any(phrase in response.lower() for phrase in self.quality_detractors['generic']):
            recommendations.append("Personalize the response to the specific complaint rather than using generic language")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _detailed_response_analysis(self, response: str) -> Dict[str, Any]:
        """Provide detailed analysis of response characteristics"""
        analysis = {}
        
        # Basic statistics
        words = response.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        
        analysis['statistics'] = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0
        }
        
        # Sentiment analysis
        blob = TextBlob(response)
        analysis['sentiment'] = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
        # Keyword analysis
        analysis['keywords'] = {
            'professional_phrases': [
                phrase for phrase in self.tone_patterns['professional']
                if phrase.lower() in response.lower()
            ],
            'empathetic_phrases': [
                phrase for phrase in self.tone_patterns['empathetic']
                if phrase.lower() in response.lower()
            ],
            'action_phrases': [
                phrase for phrase in self.quality_indicators['action_words']
                if phrase.lower() in response.lower()
            ]
        }
        
        # Issues detected
        issues = []
        if any(phrase in response.lower() for phrase in self.quality_detractors['dismissive']):
            issues.append("Dismissive language detected")
        if any(phrase in response.lower() for phrase in self.quality_detractors['unclear']):
            issues.append("Unclear language detected")
        if len(words) < 15:
            issues.append("Response may be too brief")
        
        analysis['issues'] = issues
        
        return analysis
    
    def improve_response(self, response: str, complaint: str, target_quality: float = 0.8) -> Dict[str, Any]:
        """Suggest specific improvements to reach target quality"""
        current_analysis = self.predict_response_quality(response, complaint)
        
        if current_analysis['overall_score'] >= target_quality:
            return {
                'improved_response': response,
                'improvements_made': [],
                'final_score': current_analysis['overall_score'],
                'message': 'Response already meets target quality'
            }
        
        # Generate improved response suggestions
        improved_response = response
        improvements_made = []
        
        # Add empathy if missing
        if current_analysis['component_scores'].get('empathy', 0) < 0.6:
            if not any(phrase in response.lower() for phrase in ['understand', 'sorry', 'apologize']):
                improved_response = "I understand your concern and apologize for any inconvenience. " + improved_response
                improvements_made.append("Added empathetic opening")
        
        # Add acknowledgment if missing
        if current_analysis['component_scores'].get('completeness', 0) < 0.6:
            if not any(phrase in response.lower() for phrase in ['thank you for', 'appreciate']):
                improved_response = "Thank you for bringing this matter to our attention. " + improved_response
                improvements_made.append("Added acknowledgment")
        
        # Add action items if missing
        if current_analysis['component_scores'].get('actionability', 0) < 0.6:
            if not any(phrase in response.lower() for phrase in ['will', 'within', 'contact']):
                improved_response += " We will investigate this matter and follow up with you within 2-3 business days."
                improvements_made.append("Added specific action and timeline")
        
        # Add professional closing if missing
        if not improved_response.endswith(('.', '!', '?')):
            improved_response += "."
        
        if not any(phrase in improved_response.lower() for phrase in ['regards', 'sincerely', 'best']):
            improved_response += "\n\nBest regards,\nCustomer Service Team"
            improvements_made.append("Added professional closing")
        
        # Re-analyze improved response
        final_analysis = self.predict_response_quality(improved_response, complaint)
        
        return {
            'improved_response': improved_response,
            'improvements_made': improvements_made,
            'original_score': current_analysis['overall_score'],
            'final_score': final_analysis['overall_score'],
            'target_reached': final_analysis['overall_score'] >= target_quality
        }
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            if self.quality_model:
                with open('quality_model.pkl', 'wb') as f:
                    pickle.dump(self.quality_model, f)
                    
            if self.satisfaction_model:
                with open('satisfaction_model.pkl', 'wb') as f:
                    pickle.dump(self.satisfaction_model, f)
                    
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            if os.path.exists('quality_model.pkl'):
                with open('quality_model.pkl', 'rb') as f:
                    self.quality_model = pickle.load(f)
                    
            if os.path.exists('satisfaction_model.pkl'):
                with open('satisfaction_model.pkl', 'rb') as f:
                    self.satisfaction_model = pickle.load(f)
                    
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def retrain_models(self, min_samples: int = 20) -> bool:
        """Retrain models with latest data"""
        try:
            historical_data = self._get_historical_response_data()
            
            if len(historical_data) < min_samples:
                logger.warning(f"Insufficient data for retraining (need {min_samples}, got {len(historical_data)})")
                return False
            
            # Prepare training data
            X, y_quality, y_satisfaction = self._prepare_training_data(historical_data)
            
            if len(X) == 0:
                return False
            
            # Split data for validation
            X_train, X_val, y_q_train, y_q_val, y_s_train, y_s_val = train_test_split(
                X, y_quality, y_satisfaction, test_size=0.2, random_state=42
            )
            
            # Train new models
            self.quality_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.quality_model.fit(X_train, y_q_train)
            
            self.satisfaction_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.satisfaction_model.fit(X_train, y_s_train)
            
            # Evaluate models
            q_score = r2_score(y_q_val, self.quality_model.predict(X_val))
            s_score = r2_score(y_s_val, self.satisfaction_model.predict(X_val))
            
            logger.info(f"Model retrained - Quality R²: {q_score:.3f}, Satisfaction R²: {s_score:.3f}")
            
            # Save updated models
            self._save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            return False
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        try:
            historical_data = self._get_historical_response_data()
            
            if len(historical_data) < 10 or not self.quality_model:
                return {'error': 'Insufficient data or model not trained'}
            
            X, y_quality, y_satisfaction = self._prepare_training_data(historical_data)
            
            # Make predictions
            q_pred = self.quality_model.predict(X)
            s_pred = self.satisfaction_model.predict(X)
            
            # Calculate metrics
            performance = {
                'quality_model': {
                    'r2_score': r2_score(y_quality, q_pred),
                    'rmse': np.sqrt(mean_squared_error(y_quality, q_pred)),
                    'mae': np.mean(np.abs(y_quality - q_pred))
                },
                'satisfaction_model': {
                    'r2_score': r2_score(y_satisfaction, s_pred),
                    'rmse': np.sqrt(mean_squared_error(y_satisfaction, s_pred)),
                    'mae': np.mean(np.abs(y_satisfaction - s_pred))
                },
                'data_points': len(X),
                'last_updated': datetime.now().isoformat()
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating model performance: {e}")
            return {'error': str(e)}
    
    def analyze_response_trends(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Analyze response quality trends over time"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            
            # Get responses with timestamps
            query = '''
                SELECT response, rating, resolved_at, category
                FROM grievances 
                WHERE response IS NOT NULL 
                AND rating IS NOT NULL
                AND resolved_at >= ?
                ORDER BY resolved_at
            '''
            
            df = pd.read_sql_query(query, conn, params=[cutoff_date])
            conn.close()
            
            if len(df) == 0:
                return {'error': 'No data available for the specified period'}
            
            # Analyze each response
            quality_scores = []
            timestamps = []
            categories = []
            
            for _, row in df.iterrows():
                analysis = self.predict_response_quality(row['response'], "")
                quality_scores.append(analysis['overall_score'])
                timestamps.append(row['resolved_at'])
                categories.append(row['category'])
            
            # Create analysis
            df['quality_score'] = quality_scores
            df['resolved_at'] = pd.to_datetime(df['resolved_at'])
            
            trends = {
                'overall_trend': {
                    'avg_quality': np.mean(quality_scores),
                    'quality_improvement': np.corrcoef(range(len(quality_scores)), quality_scores)[0, 1],
                    'total_responses': len(quality_scores)
                },
                'category_breakdown': {},
                'weekly_trends': {}
            }
            
            # Category breakdown
            for category in df['category'].unique():
                cat_data = df[df['category'] == category]
                trends['category_breakdown'][category] = {
                    'avg_quality': cat_data['quality_score'].mean(),
                    'avg_rating': cat_data['rating'].mean(),
                    'count': len(cat_data)
                }
            
            # Weekly trends
            df['week'] = df['resolved_at'].dt.isocalendar().week
            weekly_stats = df.groupby('week').agg({
                'quality_score': 'mean',
                'rating': 'mean'
            }).to_dict('index')
            
            trends['weekly_trends'] = weekly_stats
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing response trends: {e}")
            return {'error': str(e)}
