from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Priority keywords for different urgency levels
        self.high_priority_keywords = [
            'urgent', 'emergency', 'immediate', 'critical', 'serious', 'dangerous',
            'unsafe', 'broken', 'not working', 'failed', 'crisis', 'severe',
            'terrible', 'awful', 'horrible', 'disgusting', 'unacceptable',
            'furious', 'angry', 'frustrated', 'disappointed', 'outraged'
        ]
        
        self.medium_priority_keywords = [
            'issue', 'problem', 'concern', 'difficulty', 'trouble', 'fault',
            'error', 'bug', 'glitch', 'inconvenience', 'annoying', 'bothering'
        ]
        
        self.positive_keywords = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'perfect', 'satisfied', 'happy', 'pleased', 'appreciate', 'thankful',
            'grateful', 'love', 'like', 'enjoy', 'impressed'
        ]
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment using multiple approaches and return comprehensive results
        """
        # Clean text
        cleaned_text = self.preprocess_text(text)
        
        # TextBlob analysis
        blob = TextBlob(cleaned_text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # VADER analysis
        vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
        
        # Keyword-based priority analysis
        priority_score = self.calculate_priority_score(cleaned_text.lower())
        
        # Determine overall sentiment
        sentiment = self.determine_sentiment(textblob_polarity, vader_scores['compound'])
        
        # Determine priority (1 = highest, 3 = lowest)
        priority = self.determine_priority(sentiment, priority_score, textblob_polarity, vader_scores)
        
        return {
            'sentiment': sentiment,
            'priority': priority,
            'score': (textblob_polarity + vader_scores['compound']) / 2,  # Average sentiment score
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'vader_scores': vader_scores,
            'priority_score': priority_score,
            'confidence': self.calculate_confidence(textblob_polarity, vader_scores['compound'])
        }
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common abbreviations and informal language
        text = re.sub(r'\bu\b', 'you', text, flags=re.IGNORECASE)
        text = re.sub(r'\bur\b', 'your', text, flags=re.IGNORECASE)
        text = re.sub(r'\br\b', 'are', text, flags=re.IGNORECASE)
        
        return text
    
    def calculate_priority_score(self, text):
        """Calculate priority score based on keywords"""
        high_priority_count = sum(1 for keyword in self.high_priority_keywords if keyword in text)
        medium_priority_count = sum(1 for keyword in self.medium_priority_keywords if keyword in text)
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text)
        
        # Calculate weighted score
        priority_score = (high_priority_count * 3) + (medium_priority_count * 2) - (positive_count * 1)
        return max(0, priority_score)  # Ensure non-negative
    
    def determine_sentiment(self, textblob_polarity, vader_compound):
        """Determine overall sentiment from multiple sources"""
        # Average the scores
        avg_score = (textblob_polarity + vader_compound) / 2
        
        if avg_score <= -0.1:
            return 'negative'
        elif avg_score >= 0.1:
            return 'positive'
        else:
            return 'neutral'
    
    def determine_priority(self, sentiment, priority_score, textblob_polarity, vader_scores):
        """Determine priority based on sentiment and keyword analysis"""
        # Base priority on sentiment
        if sentiment == 'negative':
            base_priority = 1  # High priority
        elif sentiment == 'neutral':
            base_priority = 2  # Medium priority
        else:
            base_priority = 3  # Low priority (positive feedback)
        
        # Adjust based on priority keywords
        if priority_score >= 5:
            return 1  # Critical - multiple urgent keywords
        elif priority_score >= 3:
            return min(base_priority, 1)  # At least high priority
        elif priority_score >= 1:
            return min(base_priority, 2)  # At least medium priority
        
        # Adjust based on intensity of negative sentiment
        if sentiment == 'negative':
            if textblob_polarity <= -0.5 or vader_scores['compound'] <= -0.5:
                return 1  # Very negative = high priority
            elif textblob_polarity <= -0.3 or vader_scores['compound'] <= -0.3:
                return min(base_priority, 2)  # Moderately negative = medium priority
        
        return base_priority
    
    def calculate_confidence(self, textblob_polarity, vader_compound):
        """Calculate confidence in sentiment analysis"""
        # Higher confidence when both methods agree and have strong scores
        agreement = abs(textblob_polarity - vader_compound)
        strength = (abs(textblob_polarity) + abs(vader_compound)) / 2
        
        confidence = (1 - agreement) * strength
        return round(min(max(confidence, 0), 1), 2)  # Normalize to 0-1
    
    def get_sentiment_emoji(self, sentiment):
        """Get emoji representation of sentiment"""
        emoji_map = {
            'positive': '😊',
            'neutral': '😐',
            'negative': '😟'
        }
        return emoji_map.get(sentiment, '❓')
    
    def get_priority_label(self, priority):
        """Get human-readable priority label"""
        priority_map = {
            1: 'High Priority',
            2: 'Medium Priority',
            3: 'Low Priority'
        }
        return priority_map.get(priority, 'Unknown')
    
    def get_priority_color(self, priority):
        """Get color code for priority visualization"""
        color_map = {
            1: '#FF6B6B',  # Red for high priority
            2: '#FFE66D',  # Yellow for medium priority
            3: '#4ECDC4'   # Green for low priority
        }
        return color_map.get(priority, '#CCCCCC')
