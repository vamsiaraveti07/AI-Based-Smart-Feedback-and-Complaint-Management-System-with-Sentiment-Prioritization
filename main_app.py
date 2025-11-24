
import re
from textblob import TextBlob
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        self.urgency_keywords = ["urgent", "critical", "immediate", "emergency", "unsafe", "danger"]
        self.severity_keywords = ["broken", "harassed", "unsafe", "lost", "denied"]

    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        sentiment = "positive" if polarity > 0.2 else "negative" if polarity < -0.2 else "neutral"

        emotion = "angry" if any(word in text.lower() for word in ["angry", "frustrated", "annoyed"]) else \
                  "sad" if any(word in text.lower() for word in ["sad", "disappointed"]) else "neutral"

        priority = self.determine_priority(sentiment, text)
        confidence = abs(polarity)

        return {
            "sentiment": sentiment,
            "priority": priority,
            "confidence": confidence,
            "emotion": emotion
        }

    def determine_priority(self, sentiment, description):
        high_urgency = any(word in description.lower() for word in self.urgency_keywords)
        if sentiment == "negative" and high_urgency:
            return 1  # High
        elif sentiment == "negative":
            return 2  # Medium
        return 3  # Low

    def get_sentiment_emoji(self, sentiment):
        return {"positive": "😊", "neutral": "😐", "negative": "😠"}.get(sentiment, "❓")

    def get_priority_label(self, level):
        return {1: "High Priority", 2: "Medium Priority", 3: "Low Priority"}.get(level, "Unknown")

    def compute_impact_score(self, description, emotion, unresolved_count):
        keyword_score = sum([1 for word in self.severity_keywords if word in description.lower()]) / 5
        length_score = min(len(description) / 500, 1.0)
        emotion_score = 1.0 if emotion == "angry" else 0.7 if emotion == "sad" else 0.3
        unresolved_score = min(unresolved_count / 3, 1.0)
        impact_score = (0.3 * keyword_score) + (0.3 * length_score) + (0.2 * emotion_score) + (0.2 * unresolved_score)
        return round(min(impact_score, 1.0), 2)

    def learn_priority_accuracy(self, resolution_time, rating):
        return resolution_time < 72 and rating >= 4

    def detect_sentiment_shift(self, history):
        if len(history) < 2:
            return False
        return history[0] == "neutral" and history[-1] == "angry"

    def is_user_likely_satisfied(self, past_ratings):
        return np.mean(past_ratings) >= 3.5 if past_ratings else False

    def is_trusted_user(self, total_submissions, avg_rating):
        return total_submissions >= 3 and avg_rating >= 4

    def grievance_bot(self, query):
        query = query.lower()
        if "status" in query:
            return "You can check your complaint status in the 'My Grievances' section."
        elif "submit" in query:
            return "To submit a new grievance, go to the 'Submit Grievance' tab."
        elif "resolved" in query:
            return "Once resolved, your grievance will show a green checkmark."
        else:
            return "I'm here to assist you with grievance-related queries. Please rephrase if needed!"

    def is_likely_delayed(self, created_at_str):
        try:
            created_at = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
            return datetime.now() - created_at > timedelta(days=5)
        except Exception:
            return False

    def generate_heatmap_df(self, raw_data):
        df = pd.DataFrame(raw_data)  # Expecting {'Department': [], 'Category': [], 'Count': []}
        heatmap_df = df.pivot(index='Department', columns='Category', values='Count').fillna(0)
        return heatmap_df.astype(int)
