"""
Smart Auto-Routing System
Automatically routes complaints to the best-suited department or personnel
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import json
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re

logger = logging.getLogger(__name__)

class SmartRoutingSystem:
    """
    Intelligent routing system for grievances based on multiple factors
    """
    
    def __init__(self, db_path: str = 'grievance_system.db'):
        self.db_path = db_path
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Initialize department and staff data
        self._init_departments_and_staff()
        self._init_routing_rules()
        self._init_performance_metrics()
        
        # Load historical routing data for learning
        self._load_historical_performance()
    
    def _init_departments_and_staff(self):
        """Initialize department and staff configuration"""
        self.departments = {
            'academic': {
                'name': 'Academic Affairs',
                'categories': ['exam', 'grades', 'course', 'faculty', 'curriculum', 'academic'],
                'keywords': ['exam', 'test', 'grade', 'marks', 'professor', 'course', 'subject', 'syllabus'],
                'staff': ['dr_smith', 'prof_johnson', 'academic_coordinator'],
                'capacity': 20,  # Max concurrent cases
                'specializations': {
                    'dr_smith': ['exams', 'grading'],
                    'prof_johnson': ['curriculum', 'faculty_issues'],
                    'academic_coordinator': ['general_academic']
                }
            },
            'hostel': {
                'name': 'Hostel & Accommodation',
                'categories': ['hostel', 'accommodation', 'room', 'food'],
                'keywords': ['room', 'hostel', 'mess', 'food', 'accommodation', 'roommate', 'facility'],
                'staff': ['hostel_warden', 'maintenance_head', 'mess_manager'],
                'capacity': 15,
                'specializations': {
                    'hostel_warden': ['room_allocation', 'discipline'],
                    'maintenance_head': ['repairs', 'facilities'],
                    'mess_manager': ['food_quality', 'mess_timing']
                }
            },
            'infrastructure': {
                'name': 'Infrastructure & Facilities',
                'categories': ['labs', 'classrooms', 'library', 'wifi', 'internet'],
                'keywords': ['lab', 'classroom', 'library', 'wifi', 'internet', 'computer', 'equipment'],
                'staff': ['it_admin', 'lab_technician', 'librarian'],
                'capacity': 12,
                'specializations': {
                    'it_admin': ['wifi', 'internet', 'computer_issues'],
                    'lab_technician': ['equipment', 'lab_maintenance'],
                    'librarian': ['library_services', 'books']
                }
            },
            'administration': {
                'name': 'Administration',
                'categories': ['fees', 'documents', 'registration', 'other'],
                'keywords': ['fee', 'payment', 'document', 'certificate', 'registration', 'admission'],
                'staff': ['admin_officer', 'accounts_manager', 'registrar'],
                'capacity': 10,
                'specializations': {
                    'admin_officer': ['general_admin', 'documentation'],
                    'accounts_manager': ['fees', 'payments'],
                    'registrar': ['registration', 'certificates']
                }
            },
            'student_affairs': {
                'name': 'Student Affairs & Welfare',
                'categories': ['discipline', 'counseling', 'events', 'clubs'],
                'keywords': ['discipline', 'counseling', 'event', 'club', 'society', 'mental_health'],
                'staff': ['student_counselor', 'dean_students', 'activities_coordinator'],
                'capacity': 8,
                'specializations': {
                    'student_counselor': ['counseling', 'mental_health'],
                    'dean_students': ['discipline', 'serious_issues'],
                    'activities_coordinator': ['events', 'clubs']
                }
            }
        }
        
        # Staff workload and performance tracking
        self.staff_metrics = {}
        for dept_data in self.departments.values():
            for staff_id in dept_data['staff']:
                self.staff_metrics[staff_id] = {
                    'current_workload': 0,
                    'max_capacity': 5,  # Default capacity per person
                    'avg_resolution_time': 48,  # hours
                    'satisfaction_rating': 4.0,
                    'specialization_match_rate': 0.8,
                    'total_resolved': 0,
                    'last_assignment': datetime.now() - timedelta(hours=1)
                }
    
    def _init_routing_rules(self):
        """Initialize routing rules and weights"""
        self.routing_weights = {
            'keyword_match': 0.3,
            'category_match': 0.25,
            'staff_performance': 0.2,
            'workload_balance': 0.15,
            'specialization_match': 0.1
        }
        
        # Priority escalation rules
        self.escalation_rules = {
            'high_priority': {
                'route_to': 'senior_staff',
                'max_response_time': 2,  # hours
                'notify_management': True
            },
            'urgent': {
                'route_to': 'department_head',
                'max_response_time': 1,  # hours
                'notify_management': True
            }
        }
    
    def _init_performance_metrics(self):
        """Initialize performance tracking"""
        self.routing_history = []
        self.department_performance = {}
        self.staff_performance = {}
        
        # Initialize database tables for routing metrics
        self._create_routing_tables()
    
    def _create_routing_tables(self):
        """Create necessary database tables for routing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Routing history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS routing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grievance_id INTEGER,
                department TEXT,
                assigned_staff TEXT,
                routing_score REAL,
                assignment_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolution_time REAL,
                satisfaction_rating INTEGER,
                routing_accuracy REAL,
                FOREIGN KEY (grievance_id) REFERENCES grievances (id)
            )
        ''')
        
        # Staff workload table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS staff_workload (
                staff_id TEXT PRIMARY KEY,
                current_load INTEGER DEFAULT 0,
                max_capacity INTEGER DEFAULT 5,
                department TEXT,
                specializations TEXT,
                avg_resolution_time REAL DEFAULT 48.0,
                satisfaction_rating REAL DEFAULT 4.0,
                total_assignments INTEGER DEFAULT 0,
                total_resolved INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Department performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS department_performance (
                department TEXT PRIMARY KEY,
                total_cases INTEGER DEFAULT 0,
                avg_resolution_time REAL DEFAULT 48.0,
                satisfaction_rating REAL DEFAULT 4.0,
                current_capacity INTEGER DEFAULT 20,
                utilization_rate REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_historical_performance(self):
        """Load historical performance data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load staff performance
            staff_df = pd.read_sql_query('SELECT * FROM staff_workload', conn)
            for _, row in staff_df.iterrows():
                if row['staff_id'] in self.staff_metrics:
                    self.staff_metrics[row['staff_id']].update({
                        'current_workload': row['current_load'],
                        'max_capacity': row['max_capacity'],
                        'avg_resolution_time': row['avg_resolution_time'],
                        'satisfaction_rating': row['satisfaction_rating'],
                        'total_resolved': row['total_resolved']
                    })
            
            # Load department performance
            dept_df = pd.read_sql_query('SELECT * FROM department_performance', conn)
            for _, row in dept_df.iterrows():
                self.department_performance[row['department']] = {
                    'total_cases': row['total_cases'],
                    'avg_resolution_time': row['avg_resolution_time'],
                    'satisfaction_rating': row['satisfaction_rating'],
                    'utilization_rate': row['utilization_rate']
                }
            
            conn.close()
            logger.info("Historical performance data loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
    
    def route_grievance(self, grievance_data: Dict) -> Dict[str, Any]:
        """
        Main routing function - determines best department and staff
        """
        grievance_text = grievance_data.get('description', '')
        category = grievance_data.get('category', '')
        priority = grievance_data.get('priority', 3)
        emotion_data = grievance_data.get('emotion_analysis', {})
        impact_score = grievance_data.get('impact_score', 0.0)
        
        # Step 1: Determine department
        department_scores = self._score_departments(grievance_text, category, priority)
        best_department = max(department_scores.items(), key=lambda x: x[1])[0]
        
        # Step 2: Determine best staff member in department
        staff_scores = self._score_staff_members(
            best_department, grievance_text, priority, emotion_data
        )
        
        best_staff = max(staff_scores.items(), key=lambda x: x[1])[0] if staff_scores else None
        
        # Step 3: Check for escalation needs
        escalation_info = self._check_escalation_needs(priority, impact_score, emotion_data)
        
        # Step 4: Apply load balancing if needed
        final_assignment = self._apply_load_balancing(
            best_department, best_staff, priority, escalation_info
        )
        
        # Step 5: Record routing decision
        routing_result = {
            'department': final_assignment['department'],
            'assigned_staff': final_assignment['staff'],
            'routing_score': final_assignment['score'],
            'routing_reason': final_assignment['reason'],
            'escalation_applied': escalation_info['escalation_needed'],
            'estimated_resolution_time': self._estimate_resolution_time(
                final_assignment['department'], final_assignment['staff'], priority
            ),
            'confidence': self._calculate_routing_confidence(department_scores, staff_scores),
            'alternative_assignments': self._get_alternative_assignments(
                department_scores, staff_scores, 2
            )
        }
        
        # Update workload
        self._update_staff_workload(final_assignment['staff'], 1)
        
        # Log routing decision
        self._log_routing_decision(grievance_data.get('id'), routing_result)
        
        return routing_result
    
    def _score_departments(self, text: str, category: str, priority: int) -> Dict[str, float]:
        """Score departments based on text analysis and category matching"""
        scores = {}
        
        for dept_id, dept_data in self.departments.items():
            score = 0.0
            
            # Category match score
            if category.lower() in [cat.lower() for cat in dept_data['categories']]:
                score += self.routing_weights['category_match']
            
            # Keyword match score
            keyword_score = self._calculate_keyword_match(text, dept_data['keywords'])
            score += keyword_score * self.routing_weights['keyword_match']
            
            # Department performance score
            dept_perf = self.department_performance.get(dept_id, {})
            perf_score = min(dept_perf.get('satisfaction_rating', 4.0) / 5.0, 1.0)
            score += perf_score * self.routing_weights['staff_performance']
            
            # Workload balance score (higher score for less utilized departments)
            utilization = dept_perf.get('utilization_rate', 0.5)
            workload_score = 1.0 - min(utilization, 1.0)
            score += workload_score * self.routing_weights['workload_balance']
            
            scores[dept_id] = min(score, 1.0)
        
        return scores
    
    def _calculate_keyword_match(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword match score"""
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        return min(matches / len(keywords), 1.0) if keywords else 0.0
    
    def _score_staff_members(self, department: str, text: str, 
                           priority: int, emotion_data: Dict) -> Dict[str, float]:
        """Score staff members within a department"""
        scores = {}
        dept_data = self.departments.get(department, {})
        
        for staff_id in dept_data.get('staff', []):
            if staff_id not in self.staff_metrics:
                continue
                
            staff_data = self.staff_metrics[staff_id]
            score = 0.0
            
            # Performance score
            perf_score = min(staff_data['satisfaction_rating'] / 5.0, 1.0)
            score += perf_score * self.routing_weights['staff_performance']
            
            # Workload balance score
            workload_ratio = staff_data['current_workload'] / staff_data['max_capacity']
            workload_score = max(1.0 - workload_ratio, 0.0)
            score += workload_score * self.routing_weights['workload_balance']
            
            # Specialization match score
            spec_score = self._calculate_specialization_match(
                staff_id, text, dept_data.get('specializations', {})
            )
            score += spec_score * self.routing_weights['specialization_match']
            
            # Priority handling capability
            if priority == 1:  # High priority
                # Prefer experienced staff for high priority
                experience_factor = min(staff_data['total_resolved'] / 50, 1.0)
                score *= (1.0 + experience_factor * 0.3)
            
            # Emotion handling capability
            if emotion_data.get('primary_emotion') in ['anger', 'frustration']:
                # Prefer staff with good satisfaction ratings for emotional cases
                emotion_handling = min(staff_data['satisfaction_rating'] / 5.0, 1.0)
                score *= (1.0 + emotion_handling * 0.2)
            
            scores[staff_id] = min(score, 1.0)
        
        return scores
    
    def _calculate_specialization_match(self, staff_id: str, text: str, 
                                      specializations: Dict) -> float:
        """Calculate how well staff specialization matches the grievance"""
        staff_specs = specializations.get(staff_id, [])
        if not staff_specs:
            return 0.5  # Default score
        
        text_lower = text.lower()
        spec_matches = 0
        
        for spec in staff_specs:
            # Convert specialization to keywords
            spec_keywords = spec.replace('_', ' ').split()
            if any(keyword in text_lower for keyword in spec_keywords):
                spec_matches += 1
        
        return min(spec_matches / len(staff_specs), 1.0)
    
    def _check_escalation_needs(self, priority: int, impact_score: float, 
                              emotion_data: Dict) -> Dict[str, Any]:
        """Check if escalation is needed based on various factors"""
        escalation_info = {
            'escalation_needed': False,
            'escalation_level': 'none',
            'reason': '',
            'notify_management': False
        }
        
        # High priority automatic escalation
        if priority == 1:
            escalation_info.update({
                'escalation_needed': True,
                'escalation_level': 'high_priority',
                'reason': 'High priority grievance',
                'notify_management': True
            })
        
        # High impact score escalation
        elif impact_score >= 0.8:
            escalation_info.update({
                'escalation_needed': True,
                'escalation_level': 'high_impact',
                'reason': 'High impact score detected',
                'notify_management': True
            })
        
        # Emotion-based escalation
        elif emotion_data.get('primary_emotion') == 'anger' and \
             emotion_data.get('intensity', 0) > 0.7:
            escalation_info.update({
                'escalation_needed': True,
                'escalation_level': 'emotional_urgency',
                'reason': 'High emotional intensity detected',
                'notify_management': False
            })
        
        return escalation_info
    
    def _apply_load_balancing(self, department: str, staff: str, 
                            priority: int, escalation_info: Dict) -> Dict[str, Any]:
        """Apply load balancing and finalize assignment"""
        
        # Check if original assignment is valid
        if staff and self.staff_metrics[staff]['current_workload'] < \
           self.staff_metrics[staff]['max_capacity']:
            return {
                'department': department,
                'staff': staff,
                'score': 1.0,
                'reason': 'Optimal assignment based on routing algorithm'
            }
        
        # Find alternative staff in same department
        dept_data = self.departments.get(department, {})
        alternative_staff = []
        
        for staff_id in dept_data.get('staff', []):
            if staff_id != staff and \
               self.staff_metrics[staff_id]['current_workload'] < \
               self.staff_metrics[staff_id]['max_capacity']:
                alternative_staff.append(staff_id)
        
        if alternative_staff:
            # Choose best available staff
            best_alt = min(alternative_staff, 
                          key=lambda s: self.staff_metrics[s]['current_workload'])
            return {
                'department': department,
                'staff': best_alt,
                'score': 0.8,
                'reason': 'Load balancing: original staff overloaded'
            }
        
        # If department is full, find alternative department
        alternative_dept = self._find_alternative_department(department, priority)
        if alternative_dept:
            alt_staff = self._find_available_staff(alternative_dept)
            return {
                'department': alternative_dept,
                'staff': alt_staff,
                'score': 0.6,
                'reason': 'Department overloaded: routed to alternative'
            }
        
        # Last resort: assign to least loaded staff in original department
        least_loaded = min(dept_data.get('staff', []), 
                          key=lambda s: self.staff_metrics[s]['current_workload'])
        return {
            'department': department,
            'staff': least_loaded,
            'score': 0.4,
            'reason': 'System overloaded: assigned to least loaded staff'
        }
    
    def _find_alternative_department(self, original_dept: str, priority: int) -> Optional[str]:
        """Find alternative department when original is overloaded"""
        alternatives = []
        
        for dept_id, dept_data in self.departments.items():
            if dept_id == original_dept:
                continue
                
            # Check if department has capacity
            current_load = sum(
                self.staff_metrics[staff]['current_workload'] 
                for staff in dept_data['staff'] 
                if staff in self.staff_metrics
            )
            
            if current_load < dept_data['capacity']:
                alternatives.append(dept_id)
        
        # For high priority, prefer departments with better performance
        if priority == 1 and alternatives:
            return max(alternatives, key=lambda d: 
                      self.department_performance.get(d, {}).get('satisfaction_rating', 4.0))
        
        return alternatives[0] if alternatives else None
    
    def _find_available_staff(self, department: str) -> Optional[str]:
        """Find available staff in a department"""
        dept_data = self.departments.get(department, {})
        
        for staff_id in dept_data.get('staff', []):
            if staff_id in self.staff_metrics and \
               self.staff_metrics[staff_id]['current_workload'] < \
               self.staff_metrics[staff_id]['max_capacity']:
                return staff_id
        
        # Return least loaded if none available
        if dept_data.get('staff'):
            return min(dept_data['staff'], 
                      key=lambda s: self.staff_metrics.get(s, {}).get('current_workload', 0))
        
        return None
    
    def _estimate_resolution_time(self, department: str, staff: str, priority: int) -> float:
        """Estimate resolution time based on historical data"""
        base_time = self.staff_metrics.get(staff, {}).get('avg_resolution_time', 48)
        dept_time = self.department_performance.get(department, {}).get('avg_resolution_time', 48)
        
        # Average staff and department times
        estimated_time = (base_time + dept_time) / 2
        
        # Adjust for priority
        priority_multipliers = {1: 0.5, 2: 0.8, 3: 1.0}
        estimated_time *= priority_multipliers.get(priority, 1.0)
        
        # Adjust for current workload
        workload_factor = 1.0 + (self.staff_metrics.get(staff, {}).get('current_workload', 0) * 0.2)
        estimated_time *= workload_factor
        
        return round(estimated_time, 1)
    
    def _calculate_routing_confidence(self, dept_scores: Dict, staff_scores: Dict) -> float:
        """Calculate confidence in routing decision"""
        if not dept_scores:
            return 0.0
        
        # Confidence based on score distribution
        dept_values = list(dept_scores.values())
        best_dept_score = max(dept_values)
        dept_spread = np.std(dept_values) if len(dept_values) > 1 else 0
        
        dept_confidence = best_dept_score * (1.0 - dept_spread)
        
        if staff_scores:
            staff_values = list(staff_scores.values())
            best_staff_score = max(staff_values)
            staff_spread = np.std(staff_values) if len(staff_values) > 1 else 0
            staff_confidence = best_staff_score * (1.0 - staff_spread)
            
            return (dept_confidence + staff_confidence) / 2
        
        return dept_confidence
    
    def _get_alternative_assignments(self, dept_scores: Dict, staff_scores: Dict, 
                                   num_alternatives: int) -> List[Dict]:
        """Get alternative assignment options"""
        alternatives = []
        
        # Sort departments by score
        sorted_depts = sorted(dept_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (dept, score) in enumerate(sorted_depts[1:num_alternatives+1]):
            alt_staff = None
            if staff_scores:
                # Find best available staff in this department
                dept_staff = self.departments.get(dept, {}).get('staff', [])
                available_staff = [s for s in dept_staff if s in staff_scores]
                if available_staff:
                    alt_staff = max(available_staff, key=lambda s: staff_scores[s])
            
            alternatives.append({
                'department': dept,
                'staff': alt_staff,
                'score': score,
                'rank': i + 2
            })
        
        return alternatives
    
    def _update_staff_workload(self, staff_id: str, change: int):
        """Update staff workload"""
        if staff_id in self.staff_metrics:
            self.staff_metrics[staff_id]['current_workload'] += change
            self.staff_metrics[staff_id]['current_workload'] = max(
                0, self.staff_metrics[staff_id]['current_workload']
            )
            
            # Update database
            self._update_staff_workload_db(staff_id)
    
    def _update_staff_workload_db(self, staff_id: str):
        """Update staff workload in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            staff_data = self.staff_metrics[staff_id]
            cursor.execute('''
                INSERT OR REPLACE INTO staff_workload 
                (staff_id, current_load, max_capacity, avg_resolution_time, 
                 satisfaction_rating, total_resolved, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                staff_id,
                staff_data['current_workload'],
                staff_data['max_capacity'],
                staff_data['avg_resolution_time'],
                staff_data['satisfaction_rating'],
                staff_data['total_resolved'],
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error updating staff workload: {e}")
    
    def _log_routing_decision(self, grievance_id: int, routing_result: Dict):
        """Log routing decision for learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO routing_history 
                (grievance_id, department, assigned_staff, routing_score, assignment_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                grievance_id,
                routing_result['department'],
                routing_result['assigned_staff'],
                routing_result['routing_score'],
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging routing decision: {e}")
    
    def learn_from_resolution(self, grievance_id: int, resolution_data: Dict):
        """Learn from resolution outcomes to improve routing"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update routing history with resolution data
            cursor.execute('''
                UPDATE routing_history 
                SET resolution_time = ?, satisfaction_rating = ?, routing_accuracy = ?
                WHERE grievance_id = ?
            ''', (
                resolution_data.get('resolution_time_hours'),
                resolution_data.get('satisfaction_rating'),
                resolution_data.get('routing_accuracy', 0.8),
                grievance_id
            ))
            
            # Update staff performance
            staff_id = resolution_data.get('assigned_staff')
            if staff_id and staff_id in self.staff_metrics:
                self._update_staff_performance(staff_id, resolution_data)
            
            conn.commit()
            conn.close()
            
            # Adjust routing weights based on feedback
            self._adjust_routing_weights(resolution_data)
            
        except Exception as e:
            logger.error(f"Error learning from resolution: {e}")
    
    def _update_staff_performance(self, staff_id: str, resolution_data: Dict):
        """Update staff performance metrics"""
        staff_data = self.staff_metrics[staff_id]
        
        # Update average resolution time
        current_time = staff_data['avg_resolution_time']
        new_time = resolution_data.get('resolution_time_hours', current_time)
        total_resolved = staff_data['total_resolved']
        
        staff_data['avg_resolution_time'] = (
            (current_time * total_resolved + new_time) / (total_resolved + 1)
        )
        
        # Update satisfaction rating
        current_rating = staff_data['satisfaction_rating']
        new_rating = resolution_data.get('satisfaction_rating', current_rating)
        staff_data['satisfaction_rating'] = (
            (current_rating * total_resolved + new_rating) / (total_resolved + 1)
        )
        
        staff_data['total_resolved'] += 1
        staff_data['current_workload'] = max(0, staff_data['current_workload'] - 1)
        
        # Update database
        self._update_staff_workload_db(staff_id)
    
    def _adjust_routing_weights(self, resolution_data: Dict):
        """Adjust routing weights based on resolution outcomes"""
        satisfaction = resolution_data.get('satisfaction_rating', 3)
        resolution_time = resolution_data.get('resolution_time_hours', 48)
        
        # Simple weight adjustment based on outcomes
        if satisfaction >= 4 and resolution_time <= 24:
            # Good outcome - no adjustment needed
            pass
        elif satisfaction < 3:
            # Poor satisfaction - increase performance weight
            self.routing_weights['staff_performance'] = min(
                self.routing_weights['staff_performance'] * 1.1, 0.4
            )
        elif resolution_time > 72:
            # Slow resolution - increase workload balance weight
            self.routing_weights['workload_balance'] = min(
                self.routing_weights['workload_balance'] * 1.1, 0.3
            )
    
    def get_department_workload_status(self) -> Dict[str, Dict]:
        """Get current workload status for all departments"""
        status = {}
        
        for dept_id, dept_data in self.departments.items():
            current_load = sum(
                self.staff_metrics.get(staff, {}).get('current_workload', 0)
                for staff in dept_data['staff']
            )
            
            status[dept_id] = {
                'name': dept_data['name'],
                'current_load': current_load,
                'capacity': dept_data['capacity'],
                'utilization': current_load / dept_data['capacity'],
                'status': self._get_department_status(current_load, dept_data['capacity']),
                'staff_details': {
                    staff: {
                        'current_load': self.staff_metrics.get(staff, {}).get('current_workload', 0),
                        'capacity': self.staff_metrics.get(staff, {}).get('max_capacity', 5),
                        'performance': self.staff_metrics.get(staff, {}).get('satisfaction_rating', 4.0)
                    }
                    for staff in dept_data['staff']
                }
            }
        
        return status
    
    def _get_department_status(self, current_load: int, capacity: int) -> str:
        """Get department status based on workload"""
        utilization = current_load / capacity
        
        if utilization >= 0.9:
            return 'overloaded'
        elif utilization >= 0.7:
            return 'high_load'
        elif utilization >= 0.4:
            return 'normal'
        else:
            return 'low_load'
    
    def predict_routing_success(self, grievance_data: Dict) -> Dict[str, Any]:
        """Predict likelihood of successful routing"""
        routing_result = self.route_grievance(grievance_data)
        
        # Factors affecting success
        staff_id = routing_result['assigned_staff']
        department = routing_result['department']
        
        success_factors = {
            'staff_performance': self.staff_metrics.get(staff_id, {}).get('satisfaction_rating', 4.0) / 5.0,
            'workload_level': 1.0 - min(self.staff_metrics.get(staff_id, {}).get('current_workload', 0) / 5.0, 1.0),
            'department_performance': self.department_performance.get(department, {}).get('satisfaction_rating', 4.0) / 5.0,
            'routing_confidence': routing_result['confidence']
        }
        
        success_probability = np.mean(list(success_factors.values()))
        
        return {
            'success_probability': round(success_probability, 2),
            'success_factors': success_factors,
            'risk_factors': self._identify_risk_factors(routing_result, grievance_data),
            'recommendations': self._get_routing_recommendations(success_probability, success_factors)
        }
    
    def _identify_risk_factors(self, routing_result: Dict, grievance_data: Dict) -> List[str]:
        """Identify potential risk factors for resolution"""
        risks = []
        
        staff_id = routing_result['assigned_staff']
        workload = self.staff_metrics.get(staff_id, {}).get('current_workload', 0)
        
        if workload >= 4:
            risks.append("High staff workload may delay resolution")
        
        if grievance_data.get('priority') == 1 and routing_result['routing_score'] < 0.8:
            risks.append("High priority case with suboptimal routing")
        
        if routing_result.get('escalation_applied'):
            risks.append("Escalated case requires careful monitoring")
        
        emotion_data = grievance_data.get('emotion_analysis', {})
        if emotion_data.get('primary_emotion') == 'anger':
            risks.append("Angry user requires sensitive handling")
        
        return risks
    
    def _get_routing_recommendations(self, success_prob: float, factors: Dict) -> List[str]:
        """Get recommendations for improving routing success"""
        recommendations = []
        
        if success_prob < 0.7:
            recommendations.append("Consider manual review of routing assignment")
        
        if factors['workload_level'] < 0.5:
            recommendations.append("Monitor for potential delays due to high workload")
        
        if factors['staff_performance'] < 0.8:
            recommendations.append("Provide additional support to assigned staff member")
        
        if factors['routing_confidence'] < 0.6:
            recommendations.append("Low routing confidence - consider alternative assignments")
        
        return recommendations
