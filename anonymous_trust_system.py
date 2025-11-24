"""
Anonymous Complaint System with Trust Index
Balances user privacy with spam protection
"""

import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import sqlite3
import logging
import re
import json
from collections import defaultdict, Counter
import ipaddress
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

logger = logging.getLogger(__name__)

class AnonymousTrustSystem:
    """
    Anonymous complaint system with intelligent trust index for spam filtering
    """
    
    def __init__(self, db_path: str = 'grievance_system.db'):
        self.db_path = db_path
        
        # Trust scoring weights
        self.trust_weights = {
            'language_quality': 0.25,
            'content_authenticity': 0.25,
            'behavioral_patterns': 0.20,
            'temporal_patterns': 0.15,
            'ip_reputation': 0.15
        }
        
        # Spam detection thresholds
        self.spam_thresholds = {
            'min_trust_score': 0.3,
            'max_submissions_per_hour': 5,
            'max_submissions_per_day': 20,
            'min_content_length': 20,
            'max_content_length': 5000,
            'similarity_threshold': 0.85
        }
        
        # Initialize components
        self._init_trust_models()
        self._create_trust_tables()
        self._load_historical_data()
    
    def _init_trust_models(self):
        """Initialize trust assessment models"""
        # Text similarity vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        # Known spam patterns
        self.spam_patterns = [
            r'\b(viagra|cialis|casino|lottery|winner|congratulations)\b',
            r'\b(click here|free money|make money|work from home)\b',
            r'\b(limited time|act now|urgent|expires)\b',
            r'[A-Z]{5,}',  # Excessive caps
            r'(.)\1{4,}',  # Repeated characters
            r'www\.|http|\.com|\.net|\.org',  # URLs
        ]
        
        # Language quality indicators
        self.quality_indicators = {
            'positive': ['please', 'thank', 'help', 'issue', 'problem', 'concern'],
            'negative': ['urgent', 'immediately', 'terrible', 'worst', 'hate']
        }
        
        # Initialize fingerprint storage
        self.content_fingerprints = {}
        self.ip_history = defaultdict(list)
        self.user_profiles = {}
    
    def _create_trust_tables(self):
        """Create database tables for trust system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Anonymous submissions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anonymous_submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anonymous_id TEXT NOT NULL,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                trust_score REAL NOT NULL,
                spam_probability REAL NOT NULL,
                ip_hash TEXT,
                browser_fingerprint TEXT,
                language_quality REAL,
                content_authenticity REAL,
                behavioral_score REAL,
                temporal_score REAL,
                ip_reputation REAL,
                status TEXT DEFAULT 'pending_review',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP,
                reviewer_notes TEXT
            )
        ''')
        
        # Trust profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trust_profiles (
                profile_hash TEXT PRIMARY KEY,
                submission_count INTEGER DEFAULT 0,
                avg_trust_score REAL DEFAULT 0.5,
                spam_count INTEGER DEFAULT 0,
                legitimate_count INTEGER DEFAULT 0,
                last_submission TIMESTAMP,
                behavioral_patterns TEXT,
                language_patterns TEXT,
                reputation_score REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # IP reputation table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ip_reputation (
                ip_hash TEXT PRIMARY KEY,
                reputation_score REAL DEFAULT 0.5,
                submission_count INTEGER DEFAULT 0,
                spam_count INTEGER DEFAULT 0,
                legitimate_count INTEGER DEFAULT 0,
                country_code TEXT,
                is_vpn BOOLEAN DEFAULT FALSE,
                is_proxy BOOLEAN DEFAULT FALSE,
                last_seen TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Content similarity table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_similarity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL,
                similar_content_hash TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_historical_data(self):
        """Load historical trust data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load trust profiles
            profiles_df = pd.read_sql_query('SELECT * FROM trust_profiles', conn)
            for _, profile in profiles_df.iterrows():
                self.user_profiles[profile['profile_hash']] = {
                    'submission_count': profile['submission_count'],
                    'avg_trust_score': profile['avg_trust_score'],
                    'reputation_score': profile['reputation_score']
                }
            
            # Load IP reputation
            ip_df = pd.read_sql_query('SELECT * FROM ip_reputation', conn)
            for _, ip_data in ip_df.iterrows():
                self.ip_history[ip_data['ip_hash']] = {
                    'reputation_score': ip_data['reputation_score'],
                    'submission_count': ip_data['submission_count'],
                    'spam_count': ip_data['spam_count']
                }
            
            conn.close()
            logger.info("Historical trust data loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load historical trust data: {e}")
    
    def generate_anonymous_id(self, ip_address: str, user_agent: str, 
                            additional_data: Dict = None) -> str:
        """Generate anonymous ID from user characteristics"""
        # Create a consistent but anonymous identifier
        identifier_data = f"{ip_address}:{user_agent}"
        
        if additional_data:
            # Add browser fingerprinting data if available
            screen_res = additional_data.get('screen_resolution', '')
            timezone = additional_data.get('timezone', '')
            language = additional_data.get('language', '')
            identifier_data += f":{screen_res}:{timezone}:{language}"
        
        # Hash to create anonymous ID
        anonymous_id = hashlib.sha256(identifier_data.encode()).hexdigest()[:16]
        return anonymous_id
    
    def submit_anonymous_complaint(self, title: str, category: str, description: str,
                                 ip_address: str, user_agent: str = "",
                                 additional_data: Dict = None) -> Dict[str, Any]:
        """Submit anonymous complaint with trust assessment"""
        try:
            # Generate anonymous ID
            anonymous_id = self.generate_anonymous_id(ip_address, user_agent, additional_data)
            
            # Hash IP for privacy
            ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()
            
            # Create browser fingerprint
            browser_fingerprint = self._create_browser_fingerprint(user_agent, additional_data)
            
            # Calculate trust score
            trust_assessment = self._calculate_trust_score(
                title, description, anonymous_id, ip_hash, browser_fingerprint
            )
            
            # Determine if submission should be accepted
            should_accept = trust_assessment['trust_score'] >= self.spam_thresholds['min_trust_score']
            
            if should_accept:
                # Store submission
                submission_id = self._store_anonymous_submission(
                    anonymous_id, title, category, description, 
                    trust_assessment, ip_hash, browser_fingerprint
                )
                
                # Update profiles
                self._update_trust_profiles(anonymous_id, ip_hash, trust_assessment)
                
                result = {
                    'success': True,
                    'submission_id': submission_id,
                    'anonymous_id': anonymous_id,
                    'trust_score': trust_assessment['trust_score'],
                    'status': 'accepted',
                    'message': 'Your anonymous complaint has been submitted successfully.'
                }
            else:
                # Rejection due to low trust score
                result = {
                    'success': False,
                    'trust_score': trust_assessment['trust_score'],
                    'status': 'rejected',
                    'message': 'Submission rejected due to trust score. Please ensure your complaint is legitimate and detailed.',
                    'reasons': trust_assessment.get('rejection_reasons', [])
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error submitting anonymous complaint: {e}")
            return {
                'success': False,
                'status': 'error',
                'message': 'An error occurred while processing your submission.'
            }
    
    def _calculate_trust_score(self, title: str, description: str, 
                             anonymous_id: str, ip_hash: str, 
                             browser_fingerprint: str) -> Dict[str, Any]:
        """Calculate comprehensive trust score"""
        assessment = {
            'trust_score': 0.0,
            'component_scores': {},
            'rejection_reasons': [],
            'spam_probability': 0.0
        }
        
        # Language quality assessment
        lang_score = self._assess_language_quality(title + " " + description)
        assessment['component_scores']['language_quality'] = lang_score
        
        # Content authenticity assessment
        auth_score = self._assess_content_authenticity(title, description)
        assessment['component_scores']['content_authenticity'] = auth_score
        
        # Behavioral patterns assessment
        behavior_score = self._assess_behavioral_patterns(anonymous_id, description)
        assessment['component_scores']['behavioral_patterns'] = behavior_score
        
        # Temporal patterns assessment
        temporal_score = self._assess_temporal_patterns(anonymous_id, ip_hash)
        assessment['component_scores']['temporal_patterns'] = temporal_score
        
        # IP reputation assessment
        ip_score = self._assess_ip_reputation(ip_hash)
        assessment['component_scores']['ip_reputation'] = ip_score
        
        # Calculate weighted trust score
        trust_score = 0.0
        for component, weight in self.trust_weights.items():
            component_score = assessment['component_scores'].get(component, 0.5)
            trust_score += component_score * weight
        
        assessment['trust_score'] = min(max(trust_score, 0.0), 1.0)
        
        # Calculate spam probability (inverse of trust score with adjustments)
        spam_prob = 1.0 - assessment['trust_score']
        
        # Adjust based on specific red flags
        if any(re.search(pattern, description.lower()) for pattern in self.spam_patterns):
            spam_prob = min(spam_prob + 0.3, 1.0)
            assessment['rejection_reasons'].append('Spam patterns detected')
        
        if len(description) < self.spam_thresholds['min_content_length']:
            spam_prob = min(spam_prob + 0.2, 1.0)
            assessment['rejection_reasons'].append('Content too short')
        
        if len(description) > self.spam_thresholds['max_content_length']:
            spam_prob = min(spam_prob + 0.1, 1.0)
            assessment['rejection_reasons'].append('Content too long')
        
        assessment['spam_probability'] = spam_prob
        
        return assessment
    
    def _assess_language_quality(self, text: str) -> float:
        """Assess language quality and coherence"""
        try:
            # Basic metrics
            word_count = len(text.split())
            sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
            avg_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0
            
            # Initialize score
            score = 0.5
            
            # Word count assessment
            if 10 <= word_count <= 1000:
                score += 0.2
            elif word_count < 5:
                score -= 0.3
            
            # Sentence structure
            if sentence_count > 0:
                avg_words_per_sentence = word_count / sentence_count
                if 5 <= avg_words_per_sentence <= 30:
                    score += 0.1
            
            # Average word length (indicates vocabulary sophistication)
            if 4 <= avg_word_length <= 8:
                score += 0.1
            
            # Check for excessive punctuation or special characters
            special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text) if text else 0
            if special_char_ratio > 0.1:
                score -= 0.2
            
            # Check for excessive capitalization
            caps_ratio = len(re.findall(r'[A-Z]', text)) / len(text) if text else 0
            if caps_ratio > 0.3:
                score -= 0.2
            
            # Sentiment analysis for coherence
            blob = TextBlob(text)
            if abs(blob.sentiment.polarity) > 0.1:  # Has clear sentiment
                score += 0.1
            
            # Grammar and spelling (simplified check)
            words = text.split()
            if len(words) > 5:
                # Check for common misspellings or nonsense
                english_word_ratio = self._estimate_english_word_ratio(words)
                score += english_word_ratio * 0.2
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing language quality: {e}")
            return 0.5
    
    def _estimate_english_word_ratio(self, words: List[str]) -> float:
        """Estimate ratio of English words (simplified)"""
        # Simple heuristic based on common English patterns
        english_patterns = [
            r'^[a-zA-Z]+$',  # Only letters
            r'.*[aeiou].*',  # Contains vowels
            r'.{2,}',        # At least 2 characters
        ]
        
        english_count = 0
        for word in words:
            word_clean = re.sub(r'[^a-zA-Z]', '', word.lower())
            if len(word_clean) >= 2:
                if all(re.match(pattern, word_clean) for pattern in english_patterns):
                    english_count += 1
        
        return english_count / len(words) if words else 0
    
    def _assess_content_authenticity(self, title: str, description: str) -> float:
        """Assess if content appears authentic and legitimate"""
        score = 0.5
        content = title + " " + description
        
        # Check for specific grievance-related keywords
        grievance_keywords = [
            'problem', 'issue', 'concern', 'complaint', 'help', 'fix', 'resolve',
            'service', 'quality', 'staff', 'facility', 'room', 'food', 'class',
            'exam', 'grade', 'library', 'wifi', 'internet', 'hostel', 'mess'
        ]
        
        keyword_count = sum(1 for keyword in grievance_keywords 
                           if keyword in content.lower())
        
        if keyword_count >= 2:
            score += 0.3
        elif keyword_count == 1:
            score += 0.1
        else:
            score -= 0.2
        
        # Check for specific details (numbers, dates, names)
        if re.search(r'\b\d+\b', content):  # Contains numbers
            score += 0.1
        
        if re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|yesterday|last week|this week)\b', content.lower()):
            score += 0.1
        
        # Check for emotional authenticity
        emotion_words = ['frustrated', 'disappointed', 'upset', 'concerned', 
                        'worried', 'pleased', 'satisfied', 'grateful']
        if any(word in content.lower() for word in emotion_words):
            score += 0.2
        
        # Check against spam content
        commercial_words = ['buy', 'sell', 'money', 'free', 'win', 'prize', 
                           'offer', 'deal', 'discount', 'sale']
        if any(word in content.lower() for word in commercial_words):
            score -= 0.4
        
        # Check for coherent problem description
        problem_indicators = ['because', 'since', 'due to', 'caused by', 'result']
        if any(indicator in content.lower() for indicator in problem_indicators):
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _assess_behavioral_patterns(self, anonymous_id: str, description: str) -> float:
        """Assess behavioral patterns for this anonymous user"""
        score = 0.5
        
        # Check user profile if exists
        if anonymous_id in self.user_profiles:
            profile = self.user_profiles[anonymous_id]
            
            # Reward consistent users with good history
            if profile['avg_trust_score'] > 0.7:
                score += 0.3
            elif profile['avg_trust_score'] < 0.3:
                score -= 0.3
            
            # Check submission frequency
            if profile['submission_count'] > 50:  # Very frequent submitter
                score -= 0.1
        else:
            # New user - slightly positive bias for genuine new complaints
            score += 0.1
        
        # Check content similarity with recent submissions
        similarity_score = self._check_content_similarity(description)
        if similarity_score > self.spam_thresholds['similarity_threshold']:
            score -= 0.4  # Likely duplicate or spam
        elif similarity_score > 0.7:
            score -= 0.2  # Somewhat similar content
        
        return min(max(score, 0.0), 1.0)
    
    def _assess_temporal_patterns(self, anonymous_id: str, ip_hash: str) -> float:
        """Assess temporal submission patterns"""
        score = 0.5
        
        # Check submission frequency from this IP
        current_time = datetime.now()
        recent_submissions = self._get_recent_submissions(ip_hash, hours=24)
        
        if len(recent_submissions) > self.spam_thresholds['max_submissions_per_day']:
            score -= 0.4
        elif len(recent_submissions) > self.spam_thresholds['max_submissions_per_hour']:
            # Check if within the last hour
            hour_ago = current_time - timedelta(hours=1)
            recent_hour = [s for s in recent_submissions if s > hour_ago]
            if len(recent_hour) > self.spam_thresholds['max_submissions_per_hour']:
                score -= 0.3
        
        # Check for suspicious timing patterns
        if recent_submissions:
            # Calculate time intervals between submissions
            intervals = []
            for i in range(1, len(recent_submissions)):
                interval = (recent_submissions[i] - recent_submissions[i-1]).total_seconds()
                intervals.append(interval)
            
            if intervals:
                # Very regular intervals might indicate automated behavior
                interval_std = np.std(intervals)
                if interval_std < 60:  # Less than 1 minute variation
                    score -= 0.2
        
        # Check submission time (suspicious if always at odd hours)
        current_hour = current_time.hour
        if 2 <= current_hour <= 6:  # Late night submissions might be suspicious
            score -= 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _assess_ip_reputation(self, ip_hash: str) -> float:
        """Assess IP address reputation"""
        score = 0.5
        
        if ip_hash in self.ip_history:
            ip_data = self.ip_history[ip_hash]
            
            # Use existing reputation score
            score = ip_data['reputation_score']
            
            # Adjust based on spam history
            spam_ratio = ip_data['spam_count'] / max(ip_data['submission_count'], 1)
            if spam_ratio > 0.5:
                score -= 0.3
            elif spam_ratio > 0.2:
                score -= 0.1
        
        # Additional checks could include:
        # - VPN/Proxy detection
        # - Geolocation analysis
        # - Known malicious IP lists
        # For now, we'll use a simple heuristic
        
        return min(max(score, 0.0), 1.0)
    
    def _check_content_similarity(self, description: str) -> float:
        """Check similarity with recent submissions"""
        try:
            # Get recent submissions for comparison
            conn = sqlite3.connect(self.db_path)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            
            recent_descriptions = pd.read_sql_query('''
                SELECT description FROM anonymous_submissions 
                WHERE created_at >= ? 
                ORDER BY created_at DESC 
                LIMIT 100
            ''', conn, params=[recent_cutoff])
            
            conn.close()
            
            if len(recent_descriptions) == 0:
                return 0.0
            
            # Calculate similarity using TF-IDF
            all_descriptions = recent_descriptions['description'].tolist() + [description]
            
            if len(all_descriptions) < 2:
                return 0.0
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_descriptions)
            
            # Compare new description with all recent ones
            new_desc_vector = tfidf_matrix[-1]
            similarities = cosine_similarity(new_desc_vector, tfidf_matrix[:-1]).flatten()
            
            return float(np.max(similarities)) if len(similarities) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error checking content similarity: {e}")
            return 0.0
    
    def _get_recent_submissions(self, ip_hash: str, hours: int = 24) -> List[datetime]:
        """Get recent submission timestamps for an IP"""
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            submissions = pd.read_sql_query('''
                SELECT created_at FROM anonymous_submissions 
                WHERE ip_hash = ? AND created_at >= ?
                ORDER BY created_at DESC
            ''', conn, params=[ip_hash, cutoff_time])
            
            conn.close()
            
            return [datetime.fromisoformat(ts) for ts in submissions['created_at'].tolist()]
            
        except Exception as e:
            logger.error(f"Error getting recent submissions: {e}")
            return []
    
    def _create_browser_fingerprint(self, user_agent: str, 
                                  additional_data: Dict = None) -> str:
        """Create browser fingerprint for tracking"""
        fingerprint_data = user_agent
        
        if additional_data:
            fingerprint_data += json.dumps(additional_data, sort_keys=True)
        
        return hashlib.md5(fingerprint_data.encode()).hexdigest()
    
    def _store_anonymous_submission(self, anonymous_id: str, title: str, 
                                  category: str, description: str,
                                  trust_assessment: Dict, ip_hash: str,
                                  browser_fingerprint: str) -> int:
        """Store anonymous submission in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO anonymous_submissions 
                (anonymous_id, title, category, description, trust_score, spam_probability,
                 ip_hash, browser_fingerprint, language_quality, content_authenticity,
                 behavioral_score, temporal_score, ip_reputation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                anonymous_id, title, category, description,
                trust_assessment['trust_score'], trust_assessment['spam_probability'],
                ip_hash, browser_fingerprint,
                trust_assessment['component_scores']['language_quality'],
                trust_assessment['component_scores']['content_authenticity'],
                trust_assessment['component_scores']['behavioral_patterns'],
                trust_assessment['component_scores']['temporal_patterns'],
                trust_assessment['component_scores']['ip_reputation']
            ))
            
            submission_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return submission_id
            
        except Exception as e:
            logger.error(f"Error storing anonymous submission: {e}")
            return 0
    
    def _update_trust_profiles(self, anonymous_id: str, ip_hash: str, 
                             trust_assessment: Dict):
        """Update trust profiles for user and IP"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update user profile
            cursor.execute('''
                INSERT OR REPLACE INTO trust_profiles 
                (profile_hash, submission_count, avg_trust_score, last_submission, updated_at)
                VALUES (?, 
                    COALESCE((SELECT submission_count FROM trust_profiles WHERE profile_hash = ?), 0) + 1,
                    COALESCE(
                        ((SELECT avg_trust_score FROM trust_profiles WHERE profile_hash = ?) * 
                         (SELECT submission_count FROM trust_profiles WHERE profile_hash = ?) + ?) /
                        (COALESCE((SELECT submission_count FROM trust_profiles WHERE profile_hash = ?), 0) + 1),
                        ?
                    ),
                    ?, ?
                )
            ''', (
                anonymous_id, anonymous_id, anonymous_id, anonymous_id, 
                trust_assessment['trust_score'], anonymous_id, 
                trust_assessment['trust_score'], datetime.now(), datetime.now()
            ))
            
            # Update IP reputation
            cursor.execute('''
                INSERT OR REPLACE INTO ip_reputation 
                (ip_hash, reputation_score, submission_count, last_seen)
                VALUES (?, 
                    COALESCE(
                        ((SELECT reputation_score FROM ip_reputation WHERE ip_hash = ?) * 
                         (SELECT submission_count FROM ip_reputation WHERE ip_hash = ?) + ?) /
                        (COALESCE((SELECT submission_count FROM ip_reputation WHERE ip_hash = ?), 0) + 1),
                        ?
                    ),
                    COALESCE((SELECT submission_count FROM ip_reputation WHERE ip_hash = ?), 0) + 1,
                    ?
                )
            ''', (
                ip_hash, ip_hash, ip_hash, trust_assessment['trust_score'],
                ip_hash, trust_assessment['trust_score'], ip_hash, datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating trust profiles: {e}")
    
    def review_submission(self, submission_id: int, is_legitimate: bool, 
                         reviewer_notes: str = "") -> bool:
        """Admin review of submission to improve trust model"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get submission details
            cursor.execute('''
                SELECT anonymous_id, ip_hash, trust_score, spam_probability 
                FROM anonymous_submissions WHERE id = ?
            ''', (submission_id,))
            
            submission = cursor.fetchone()
            if not submission:
                return False
            
            anonymous_id, ip_hash, trust_score, spam_probability = submission
            
            # Update submission status
            new_status = 'approved' if is_legitimate else 'spam'
            cursor.execute('''
                UPDATE anonymous_submissions 
                SET status = ?, reviewed_at = ?, reviewer_notes = ?
                WHERE id = ?
            ''', (new_status, datetime.now(), reviewer_notes, submission_id))
            
            # Update trust profiles based on review
            if is_legitimate:
                # Increase trust for this user and IP
                cursor.execute('''
                    UPDATE trust_profiles 
                    SET legitimate_count = legitimate_count + 1,
                        reputation_score = MIN(reputation_score + 0.1, 1.0)
                    WHERE profile_hash = ?
                ''', (anonymous_id,))
                
                cursor.execute('''
                    UPDATE ip_reputation 
                    SET legitimate_count = legitimate_count + 1,
                        reputation_score = MIN(reputation_score + 0.1, 1.0)
                    WHERE ip_hash = ?
                ''', (ip_hash,))
            else:
                # Decrease trust for spam
                cursor.execute('''
                    UPDATE trust_profiles 
                    SET spam_count = spam_count + 1,
                        reputation_score = MAX(reputation_score - 0.2, 0.0)
                    WHERE profile_hash = ?
                ''', (anonymous_id,))
                
                cursor.execute('''
                    UPDATE ip_reputation 
                    SET spam_count = spam_count + 1,
                        reputation_score = MAX(reputation_score - 0.2, 0.0)
                    WHERE ip_hash = ?
                ''', (ip_hash,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error reviewing submission: {e}")
            return False
    
    def get_pending_reviews(self, limit: int = 50) -> List[Dict]:
        """Get submissions pending review"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            pending_df = pd.read_sql_query('''
                SELECT id, anonymous_id, title, category, description, 
                       trust_score, spam_probability, created_at
                FROM anonymous_submissions 
                WHERE status = 'pending_review'
                ORDER BY spam_probability DESC, created_at ASC
                LIMIT ?
            ''', conn, params=[limit])
            
            conn.close()
            
            return pending_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting pending reviews: {e}")
            return []
    
    def get_trust_analytics(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Get analytics on trust system performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            
            # Submission statistics
            stats_df = pd.read_sql_query('''
                SELECT status, COUNT(*) as count, AVG(trust_score) as avg_trust,
                       AVG(spam_probability) as avg_spam_prob
                FROM anonymous_submissions 
                WHERE created_at >= ?
                GROUP BY status
            ''', conn, params=[cutoff_date])
            
            # Trust score distribution
            trust_df = pd.read_sql_query('''
                SELECT trust_score, spam_probability, status
                FROM anonymous_submissions 
                WHERE created_at >= ?
            ''', conn, params=[cutoff_date])
            
            # IP reputation stats
            ip_df = pd.read_sql_query('''
                SELECT reputation_score, submission_count, spam_count
                FROM ip_reputation
            ''', conn)
            
            conn.close()
            
            analytics = {
                'submission_stats': stats_df.to_dict('records'),
                'total_submissions': len(trust_df),
                'avg_trust_score': trust_df['trust_score'].mean() if len(trust_df) > 0 else 0,
                'trust_score_distribution': trust_df['trust_score'].hist(bins=10).tolist() if len(trust_df) > 0 else [],
                'spam_detection_accuracy': self._calculate_detection_accuracy(trust_df),
                'ip_reputation_summary': {
                    'total_ips': len(ip_df),
                    'avg_reputation': ip_df['reputation_score'].mean() if len(ip_df) > 0 else 0.5,
                    'high_reputation_ips': len(ip_df[ip_df['reputation_score'] > 0.8]) if len(ip_df) > 0 else 0,
                    'low_reputation_ips': len(ip_df[ip_df['reputation_score'] < 0.3]) if len(ip_df) > 0 else 0
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting trust analytics: {e}")
            return {}
    
    def _calculate_detection_accuracy(self, trust_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate spam detection accuracy"""
        if len(trust_df) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        # Consider submissions with trust_score < 0.3 as predicted spam
        predicted_spam = trust_df['trust_score'] < 0.3
        actual_spam = trust_df['status'] == 'spam'
        
        # Calculate metrics
        true_positives = ((predicted_spam) & (actual_spam)).sum()
        false_positives = ((predicted_spam) & (~actual_spam)).sum()
        false_negatives = ((~predicted_spam) & (actual_spam)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def update_trust_thresholds(self, new_thresholds: Dict[str, float]):
        """Update trust scoring thresholds"""
        for key, value in new_thresholds.items():
            if key in self.spam_thresholds:
                self.spam_thresholds[key] = value
        
        logger.info(f"Trust thresholds updated: {new_thresholds}")
    
    def export_trust_data(self, filepath: str):
        """Export trust system data for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Export all relevant tables
            submissions_df = pd.read_sql_query('SELECT * FROM anonymous_submissions', conn)
            profiles_df = pd.read_sql_query('SELECT * FROM trust_profiles', conn)
            ip_df = pd.read_sql_query('SELECT * FROM ip_reputation', conn)
            
            conn.close()
            
            with pd.ExcelWriter(filepath) as writer:
                submissions_df.to_excel(writer, sheet_name='Submissions', index=False)
                profiles_df.to_excel(writer, sheet_name='Trust_Profiles', index=False)
                ip_df.to_excel(writer, sheet_name='IP_Reputation', index=False)
            
            logger.info(f"Trust data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting trust data: {e}")
