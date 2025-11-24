"""
Organizational Mood Tracker
Visual dashboards showing emotional trends across departments over time
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from collections import defaultdict, Counter
import streamlit as st

logger = logging.getLogger(__name__)

class OrganizationalMoodTracker:
    """
    Comprehensive mood tracking system for organizational insights
    """
    
    def __init__(self, db_path: str = 'grievance_system.db'):
        self.db_path = db_path
        
        # Mood categories and scoring
        self.mood_categories = {
            'satisfaction': {
                'positive': ['satisfied', 'happy', 'pleased', 'grateful', 'excellent', 'great'],
                'negative': ['unsatisfied', 'unhappy', 'disappointed', 'frustrated', 'terrible', 'awful']
            },
            'stress': {
                'high': ['stressed', 'overwhelmed', 'pressure', 'deadline', 'urgent', 'crisis'],
                'low': ['calm', 'relaxed', 'manageable', 'peaceful', 'comfortable']
            },
            'engagement': {
                'high': ['engaged', 'motivated', 'enthusiastic', 'interested', 'active'],
                'low': ['disengaged', 'unmotivated', 'bored', 'passive', 'indifferent']
            },
            'trust': {
                'high': ['trust', 'confident', 'reliable', 'dependable', 'supportive'],
                'low': ['distrust', 'suspicious', 'unreliable', 'unsupportive', 'doubt']
            }
        }
        
        # Department mapping
        self.department_mapping = {
            'academic': 'Academic Affairs',
            'hostel': 'Student Housing',
            'infrastructure': 'IT & Infrastructure', 
            'administration': 'Administration',
            'other': 'General Services'
        }
        
        # Emotional intensity weights
        self.emotion_weights = {
            'joy': 1.0,
            'optimism': 0.8,
            'trust': 0.6,
            'anticipation': 0.4,
            'neutral': 0.0,
            'surprise': 0.2,
            'fear': -0.6,
            'sadness': -0.8,
            'anger': -1.0,
            'disgust': -0.7
        }
        
        # Create mood tracking tables
        self._create_mood_tables()
    
    def _create_mood_tables(self):
        """Create database tables for mood tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Mood snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mood_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                department TEXT NOT NULL,
                time_period TEXT NOT NULL,
                mood_score REAL NOT NULL,
                satisfaction_score REAL,
                stress_score REAL,
                engagement_score REAL,
                trust_score REAL,
                total_complaints INTEGER,
                positive_sentiment_ratio REAL,
                negative_sentiment_ratio REAL,
                avg_resolution_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Mood events table (for significant mood changes)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mood_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                department TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                mood_change REAL NOT NULL,
                related_complaints INTEGER,
                date_detected TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Department benchmarks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS department_benchmarks (
                department TEXT PRIMARY KEY,
                baseline_mood REAL DEFAULT 0.0,
                target_mood REAL DEFAULT 0.5,
                satisfaction_target REAL DEFAULT 0.7,
                stress_threshold REAL DEFAULT 0.6,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_department_mood(self, department: str, time_period_days: int = 7) -> Dict[str, Any]:
        """Calculate comprehensive mood metrics for a department"""
        try:
            # Get grievances for the department and time period
            grievances = self._get_department_grievances(department, time_period_days)
            
            if not grievances:
                return {
                    'department': department,
                    'mood_score': 0.0,
                    'satisfaction_score': 0.0,
                    'stress_score': 0.0,
                    'engagement_score': 0.0,
                    'trust_score': 0.0,
                    'total_complaints': 0,
                    'insights': ['No data available for this period']
                }
            
            # Calculate mood components
            mood_analysis = {
                'department': department,
                'time_period': time_period_days,
                'total_complaints': len(grievances),
                'mood_score': 0.0,
                'satisfaction_score': 0.0,
                'stress_score': 0.0,
                'engagement_score': 0.0,
                'trust_score': 0.0,
                'sentiment_distribution': {},
                'priority_distribution': {},
                'resolution_metrics': {},
                'insights': []
            }
            
            # Analyze sentiment distribution
            sentiments = [g['sentiment'] for g in grievances]
            sentiment_counts = Counter(sentiments)
            total_grievances = len(grievances)
            
            mood_analysis['sentiment_distribution'] = {
                'positive': sentiment_counts.get('positive', 0) / total_grievances,
                'neutral': sentiment_counts.get('neutral', 0) / total_grievances,
                'negative': sentiment_counts.get('negative', 0) / total_grievances
            }
            
            # Calculate overall mood score based on sentiment
            mood_score = (
                mood_analysis['sentiment_distribution']['positive'] * 1.0 +
                mood_analysis['sentiment_distribution']['neutral'] * 0.0 +
                mood_analysis['sentiment_distribution']['negative'] * -1.0
            )
            mood_analysis['mood_score'] = mood_score
            
            # Analyze specific mood components
            mood_analysis['satisfaction_score'] = self._calculate_satisfaction_score(grievances)
            mood_analysis['stress_score'] = self._calculate_stress_score(grievances)
            mood_analysis['engagement_score'] = self._calculate_engagement_score(grievances)
            mood_analysis['trust_score'] = self._calculate_trust_score(grievances)
            
            # Priority distribution
            priorities = [g['priority'] for g in grievances]
            priority_counts = Counter(priorities)
            mood_analysis['priority_distribution'] = {
                'high': priority_counts.get(1, 0) / total_grievances,
                'medium': priority_counts.get(2, 0) / total_grievances,
                'low': priority_counts.get(3, 0) / total_grievances
            }
            
            # Resolution metrics
            mood_analysis['resolution_metrics'] = self._calculate_resolution_metrics(grievances)
            
            # Generate insights
            mood_analysis['insights'] = self._generate_mood_insights(mood_analysis)
            
            # Store snapshot
            self._store_mood_snapshot(mood_analysis)
            
            return mood_analysis
            
        except Exception as e:
            logger.error(f"Error calculating department mood: {e}")
            return {'error': str(e)}
    
    def _get_department_grievances(self, department: str, time_period_days: int) -> List[Dict]:
        """Get grievances for a specific department and time period"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            
            # Map department to category
            category = department.lower() if department.lower() in ['academic', 'hostel', 'infrastructure', 'administration'] else None
            
            if category:
                query = '''
                    SELECT g.id, g.title, g.description, g.category, g.sentiment, 
                           g.priority, g.status, g.created_at, g.resolved_at, g.rating, g.feedback
                    FROM grievances g
                    WHERE g.category = ? AND g.created_at >= ?
                    ORDER BY g.created_at DESC
                '''
                params = [category, cutoff_date]
            else:
                # Get all grievances if department not specified
                query = '''
                    SELECT g.id, g.title, g.description, g.category, g.sentiment, 
                           g.priority, g.status, g.created_at, g.resolved_at, g.rating, g.feedback
                    FROM grievances g
                    WHERE g.created_at >= ?
                    ORDER BY g.created_at DESC
                '''
                params = [cutoff_date]
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting department grievances: {e}")
            return []
    
    def _calculate_satisfaction_score(self, grievances: List[Dict]) -> float:
        """Calculate satisfaction score from grievances"""
        if not grievances:
            return 0.0
        
        # Use ratings if available
        rated_grievances = [g for g in grievances if g.get('rating') is not None]
        if rated_grievances:
            avg_rating = np.mean([g['rating'] for g in rated_grievances])
            return (avg_rating - 1) / 4  # Normalize 1-5 scale to 0-1
        
        # Fall back to sentiment and keyword analysis
        satisfaction_score = 0.0
        for grievance in grievances:
            description = grievance.get('description', '').lower()
            
            # Check for satisfaction keywords
            positive_count = sum(1 for word in self.mood_categories['satisfaction']['positive']
                               if word in description)
            negative_count = sum(1 for word in self.mood_categories['satisfaction']['negative']
                               if word in description)
            
            if positive_count > negative_count:
                satisfaction_score += 0.8
            elif negative_count > positive_count:
                satisfaction_score -= 0.8
            else:
                # Use sentiment
                if grievance.get('sentiment') == 'positive':
                    satisfaction_score += 0.6
                elif grievance.get('sentiment') == 'negative':
                    satisfaction_score -= 0.6
        
        return min(max(satisfaction_score / len(grievances), -1.0), 1.0)
    
    def _calculate_stress_score(self, grievances: List[Dict]) -> float:
        """Calculate stress level from grievances"""
        if not grievances:
            return 0.0
        
        stress_score = 0.0
        for grievance in grievances:
            description = grievance.get('description', '').lower()
            
            # Check for stress indicators
            high_stress_count = sum(1 for word in self.mood_categories['stress']['high']
                                  if word in description)
            low_stress_count = sum(1 for word in self.mood_categories['stress']['low']
                                 if word in description)
            
            # High priority complaints indicate stress
            if grievance.get('priority') == 1:
                stress_score += 0.8
            elif grievance.get('priority') == 2:
                stress_score += 0.4
            
            # Keyword-based stress
            if high_stress_count > low_stress_count:
                stress_score += 0.6
            elif low_stress_count > high_stress_count:
                stress_score -= 0.3
            
            # Unresolved complaints add stress
            if grievance.get('status') == 'Pending':
                days_pending = self._days_since_created(grievance.get('created_at'))
                if days_pending > 7:
                    stress_score += 0.5
                elif days_pending > 3:
                    stress_score += 0.3
        
        return min(max(stress_score / len(grievances), 0.0), 1.0)
    
    def _calculate_engagement_score(self, grievances: List[Dict]) -> float:
        """Calculate engagement level from grievances"""
        if not grievances:
            return 0.0
        
        engagement_score = 0.0
        total_feedback_length = 0
        
        for grievance in grievances:
            description = grievance.get('description', '').lower()
            feedback = grievance.get('feedback', '') or ''
            
            # Detailed descriptions indicate engagement
            if len(description.split()) > 50:
                engagement_score += 0.6
            elif len(description.split()) > 20:
                engagement_score += 0.3
            
            # Feedback provided indicates engagement
            if feedback and len(feedback.strip()) > 10:
                engagement_score += 0.4
                total_feedback_length += len(feedback.split())
            
            # Check for engagement keywords
            high_engagement = sum(1 for word in self.mood_categories['engagement']['high']
                                if word in description)
            low_engagement = sum(1 for word in self.mood_categories['engagement']['low']
                               if word in description)
            
            if high_engagement > low_engagement:
                engagement_score += 0.5
            elif low_engagement > high_engagement:
                engagement_score -= 0.5
        
        # Bonus for detailed feedback
        if grievances:
            avg_feedback_length = total_feedback_length / len(grievances)
            if avg_feedback_length > 10:
                engagement_score += 0.2
        
        return min(max(engagement_score / len(grievances), 0.0), 1.0)
    
    def _calculate_trust_score(self, grievances: List[Dict]) -> float:
        """Calculate trust level from grievances"""
        if not grievances:
            return 0.0
        
        trust_score = 0.0
        
        for grievance in grievances:
            description = grievance.get('description', '').lower()
            
            # Check for trust keywords
            high_trust = sum(1 for word in self.mood_categories['trust']['high']
                           if word in description)
            low_trust = sum(1 for word in self.mood_categories['trust']['low']
                          if word in description)
            
            if high_trust > low_trust:
                trust_score += 0.6
            elif low_trust > high_trust:
                trust_score -= 0.8
            
            # High ratings indicate trust
            rating = grievance.get('rating')
            if rating:
                if rating >= 4:
                    trust_score += 0.7
                elif rating <= 2:
                    trust_score -= 0.7
            
            # Quick resolution builds trust
            if grievance.get('status') == 'Resolved':
                resolution_time = self._calculate_resolution_time(grievance)
                if resolution_time and resolution_time <= 2:  # Resolved within 2 days
                    trust_score += 0.4
                elif resolution_time and resolution_time > 7:  # Took more than a week
                    trust_score -= 0.3
        
        return min(max(trust_score / len(grievances), -1.0), 1.0)
    
    def _calculate_resolution_metrics(self, grievances: List[Dict]) -> Dict[str, Any]:
        """Calculate resolution-related metrics"""
        if not grievances:
            return {}
        
        resolved_grievances = [g for g in grievances if g.get('status') == 'Resolved']
        pending_grievances = [g for g in grievances if g.get('status') == 'Pending']
        
        metrics = {
            'resolution_rate': len(resolved_grievances) / len(grievances),
            'avg_resolution_time': 0.0,
            'pending_count': len(pending_grievances),
            'overdue_count': 0
        }
        
        # Calculate average resolution time
        resolution_times = []
        for grievance in resolved_grievances:
            res_time = self._calculate_resolution_time(grievance)
            if res_time:
                resolution_times.append(res_time)
        
        if resolution_times:
            metrics['avg_resolution_time'] = np.mean(resolution_times)
        
        # Count overdue grievances (pending > 7 days)
        for grievance in pending_grievances:
            days_pending = self._days_since_created(grievance.get('created_at'))
            if days_pending > 7:
                metrics['overdue_count'] += 1
        
        return metrics
    
    def _generate_mood_insights(self, mood_analysis: Dict) -> List[str]:
        """Generate insights based on mood analysis"""
        insights = []
        
        # Overall mood insights
        mood_score = mood_analysis['mood_score']
        if mood_score > 0.5:
            insights.append("✅ Department mood is positive - users are generally satisfied")
        elif mood_score < -0.3:
            insights.append("⚠️ Department mood is concerning - multiple negative experiences detected")
        else:
            insights.append("ℹ️ Department mood is neutral - room for improvement")
        
        # Satisfaction insights
        satisfaction = mood_analysis['satisfaction_score']
        if satisfaction > 0.6:
            insights.append("😊 High satisfaction levels - keep up the good work!")
        elif satisfaction < 0.2:
            insights.append("😟 Low satisfaction detected - immediate attention needed")
        
        # Stress insights
        stress = mood_analysis['stress_score']
        if stress > 0.7:
            insights.append("🔥 High stress levels detected - workload or pressure issues")
        elif stress > 0.5:
            insights.append("⚡ Moderate stress levels - monitor for escalation")
        
        # Resolution insights
        resolution_rate = mood_analysis['resolution_metrics'].get('resolution_rate', 0)
        if resolution_rate < 0.5:
            insights.append("⏰ Low resolution rate - many complaints remain unresolved")
        
        overdue_count = mood_analysis['resolution_metrics'].get('overdue_count', 0)
        if overdue_count > 0:
            insights.append(f"📅 {overdue_count} overdue complaints need immediate attention")
        
        # Priority distribution insights
        high_priority_ratio = mood_analysis['priority_distribution'].get('high', 0)
        if high_priority_ratio > 0.4:
            insights.append("🚨 High proportion of urgent complaints - systemic issues possible")
        
        return insights[:5]  # Limit to top 5 insights
    
    def _days_since_created(self, created_at: str) -> int:
        """Calculate days since grievance was created"""
        try:
            created_date = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            return (datetime.now() - created_date).days
        except:
            return 0
    
    def _calculate_resolution_time(self, grievance: Dict) -> Optional[float]:
        """Calculate resolution time in days"""
        try:
            if not grievance.get('resolved_at') or not grievance.get('created_at'):
                return None
            
            created = datetime.strptime(grievance['created_at'], '%Y-%m-%d %H:%M:%S')
            resolved = datetime.strptime(grievance['resolved_at'], '%Y-%m-%d %H:%M:%S')
            
            return (resolved - created).total_seconds() / (24 * 3600)  # Days
        except:
            return None
    
    def _store_mood_snapshot(self, mood_analysis: Dict):
        """Store mood snapshot in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO mood_snapshots 
                (department, time_period, mood_score, satisfaction_score, stress_score,
                 engagement_score, trust_score, total_complaints, positive_sentiment_ratio,
                 negative_sentiment_ratio, avg_resolution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                mood_analysis['department'],
                f"{mood_analysis['time_period']}_days",
                mood_analysis['mood_score'],
                mood_analysis['satisfaction_score'],
                mood_analysis['stress_score'],
                mood_analysis['engagement_score'],
                mood_analysis['trust_score'],
                mood_analysis['total_complaints'],
                mood_analysis['sentiment_distribution'].get('positive', 0),
                mood_analysis['sentiment_distribution'].get('negative', 0),
                mood_analysis['resolution_metrics'].get('avg_resolution_time', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing mood snapshot: {e}")
    
    def get_mood_trends(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Get mood trends across all departments"""
        trends = {}
        
        departments = ['academic', 'hostel', 'infrastructure', 'administration']
        
        for department in departments:
            dept_mood = self.calculate_department_mood(department, time_period_days)
            trends[department] = dept_mood
        
        # Calculate overall organizational mood
        all_scores = [dept['mood_score'] for dept in trends.values() if 'mood_score' in dept]
        if all_scores:
            trends['overall'] = {
                'mood_score': np.mean(all_scores),
                'satisfaction_score': np.mean([dept['satisfaction_score'] for dept in trends.values() if 'satisfaction_score' in dept]),
                'stress_score': np.mean([dept['stress_score'] for dept in trends.values() if 'stress_score' in dept]),
                'total_complaints': sum([dept['total_complaints'] for dept in trends.values() if 'total_complaints' in dept])
            }
        
        return trends
    
    def detect_mood_events(self, threshold: float = 0.3) -> List[Dict]:
        """Detect significant mood changes or events"""
        try:
            events = []
            
            # Get recent mood snapshots
            conn = sqlite3.connect(self.db_path)
            
            # Compare current mood with previous period
            for department in ['academic', 'hostel', 'infrastructure', 'administration']:
                query = '''
                    SELECT mood_score, stress_score, satisfaction_score, created_at
                    FROM mood_snapshots 
                    WHERE department = ?
                    ORDER BY created_at DESC
                    LIMIT 5
                '''
                
                df = pd.read_sql_query(query, conn, params=[department])
                
                if len(df) >= 2:
                    current_mood = df.iloc[0]['mood_score']
                    previous_mood = df.iloc[1]['mood_score']
                    mood_change = current_mood - previous_mood
                    
                    if abs(mood_change) >= threshold:
                        event_type = 'mood_improvement' if mood_change > 0 else 'mood_deterioration'
                        severity = 'high' if abs(mood_change) > 0.5 else 'medium'
                        
                        events.append({
                            'department': department,
                            'event_type': event_type,
                            'severity': severity,
                            'mood_change': mood_change,
                            'description': f"Significant mood change detected in {department}",
                            'current_mood': current_mood,
                            'previous_mood': previous_mood
                        })
            
            conn.close()
            return events
            
        except Exception as e:
            logger.error(f"Error detecting mood events: {e}")
            return []
    
    def create_mood_dashboard(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Create comprehensive mood dashboard data"""
        try:
            # Get mood trends
            mood_trends = self.get_mood_trends(time_period_days)
            
            # Detect mood events
            mood_events = self.detect_mood_events()
            
            # Create dashboard data structure
            dashboard_data = {
                'mood_trends': mood_trends,
                'mood_events': mood_events,
                'charts': self._create_mood_charts(mood_trends),
                'alerts': self._generate_mood_alerts(mood_trends, mood_events),
                'recommendations': self._generate_mood_recommendations(mood_trends)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error creating mood dashboard: {e}")
            return {'error': str(e)}
    
    def _create_mood_charts(self, mood_trends: Dict) -> Dict[str, Any]:
        """Create chart data for mood visualization"""
        charts = {}
        
        # Department mood comparison
        departments = [dept for dept in mood_trends.keys() if dept != 'overall']
        mood_scores = [mood_trends[dept]['mood_score'] for dept in departments]
        
        charts['department_comparison'] = {
            'type': 'bar',
            'data': {
                'departments': departments,
                'mood_scores': mood_scores,
                'colors': ['green' if score > 0.3 else 'orange' if score > -0.3 else 'red' for score in mood_scores]
            }
        }
        
        # Mood components radar chart
        if 'overall' in mood_trends:
            overall = mood_trends['overall']
            charts['mood_components'] = {
                'type': 'radar',
                'data': {
                    'categories': ['Satisfaction', 'Stress', 'Engagement', 'Trust'],
                    'values': [
                        overall.get('satisfaction_score', 0),
                        1 - overall.get('stress_score', 0),  # Invert stress for radar
                        overall.get('engagement_score', 0),
                        overall.get('trust_score', 0)
                    ]
                }
            }
        
        # Sentiment distribution
        sentiment_data = {}
        for dept, data in mood_trends.items():
            if dept != 'overall' and 'sentiment_distribution' in data:
                sentiment_data[dept] = data['sentiment_distribution']
        
        charts['sentiment_distribution'] = {
            'type': 'stacked_bar',
            'data': sentiment_data
        }
        
        return charts
    
    def _generate_mood_alerts(self, mood_trends: Dict, mood_events: List[Dict]) -> List[Dict]:
        """Generate mood-based alerts"""
        alerts = []
        
        # Critical mood alerts
        for dept, data in mood_trends.items():
            if dept != 'overall' and isinstance(data, dict):
                mood_score = data.get('mood_score', 0)
                stress_score = data.get('stress_score', 0)
                
                if mood_score < -0.5:
                    alerts.append({
                        'type': 'critical',
                        'department': dept,
                        'title': f'Critical mood in {dept.title()}',
                        'message': f'Mood score is very low ({mood_score:.2f}). Immediate intervention recommended.',
                        'severity': 'high'
                    })
                
                if stress_score > 0.8:
                    alerts.append({
                        'type': 'warning',
                        'department': dept,
                        'title': f'High stress in {dept.title()}',
                        'message': f'Stress levels are concerning ({stress_score:.2f}). Consider workload review.',
                        'severity': 'medium'
                    })
        
        # Event-based alerts
        for event in mood_events:
            if event['severity'] == 'high':
                alerts.append({
                    'type': 'event',
                    'department': event['department'],
                    'title': f"Significant mood change in {event['department'].title()}",
                    'message': f"Mood change: {event['mood_change']:.2f}",
                    'severity': 'high' if event['event_type'] == 'mood_deterioration' else 'low'
                })
        
        return alerts[:10]  # Limit to top 10 alerts
    
    def _generate_mood_recommendations(self, mood_trends: Dict) -> List[Dict]:
        """Generate recommendations based on mood analysis"""
        recommendations = []
        
        for dept, data in mood_trends.items():
            if dept != 'overall' and isinstance(data, dict):
                mood_score = data.get('mood_score', 0)
                satisfaction_score = data.get('satisfaction_score', 0)
                stress_score = data.get('stress_score', 0)
                
                if satisfaction_score < 0.3:
                    recommendations.append({
                        'department': dept,
                        'type': 'satisfaction',
                        'priority': 'high',
                        'title': f'Improve satisfaction in {dept.title()}',
                        'description': 'Consider service quality review and staff training',
                        'actions': [
                            'Review complaint resolution processes',
                            'Conduct user satisfaction survey',
                            'Implement service improvements'
                        ]
                    })
                
                if stress_score > 0.6:
                    recommendations.append({
                        'department': dept,
                        'type': 'stress',
                        'priority': 'medium',
                        'title': f'Address stress in {dept.title()}',
                        'description': 'High stress levels detected, consider workload management',
                        'actions': [
                            'Review workload distribution',
                            'Implement stress reduction programs',
                            'Improve work-life balance initiatives'
                        ]
                    })
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def export_mood_report(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Export comprehensive mood report"""
        try:
            dashboard_data = self.create_mood_dashboard(time_period_days)
            
            # Add executive summary
            mood_trends = dashboard_data.get('mood_trends', {})
            overall_mood = mood_trends.get('overall', {})
            
            executive_summary = {
                'period': f"Last {time_period_days} days",
                'overall_mood': overall_mood.get('mood_score', 0),
                'total_complaints': overall_mood.get('total_complaints', 0),
                'departments_analyzed': len([d for d in mood_trends.keys() if d != 'overall']),
                'critical_alerts': len([a for a in dashboard_data.get('alerts', []) if a.get('severity') == 'high']),
                'key_insights': []
            }
            
            # Generate key insights
            if overall_mood.get('mood_score', 0) > 0.3:
                executive_summary['key_insights'].append("Overall organizational mood is positive")
            elif overall_mood.get('mood_score', 0) < -0.3:
                executive_summary['key_insights'].append("Overall organizational mood needs attention")
            
            # Add department-specific insights
            for dept, data in mood_trends.items():
                if dept != 'overall' and isinstance(data, dict):
                    if data.get('mood_score', 0) < -0.3:
                        executive_summary['key_insights'].append(f"{dept.title()} department shows concerning mood trends")
            
            dashboard_data['executive_summary'] = executive_summary
            dashboard_data['generated_at'] = datetime.now().isoformat()
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error exporting mood report: {e}")
            return {'error': str(e)}
