import sqlite3
import bcrypt
import datetime
from typing import List, Dict, Any

class DatabaseManager:
    def __init__(self, db_name='grievance_system.db'):
        self.db_name = db_name
        self.init_database()
    
    def get_connection(self):
        return sqlite3.connect(self.db_name)
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'student',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Grievances table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS grievances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                priority INTEGER NOT NULL,
                status TEXT DEFAULT 'Pending',
                assigned_to TEXT,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                response TEXT,
                rating INTEGER,
                feedback TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Notifications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                grievance_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                is_read BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (grievance_id) REFERENCES grievances (id)
            )
        ''')
        
        # Create default admin user
        self.create_default_admin()
        
        conn.commit()
        conn.close()
    
    def create_default_admin(self):
        """Create default admin user if not exists"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',))
        if not cursor.fetchone():
            password_hash = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            ''', ('admin', 'admin@system.com', password_hash, 'admin'))
        
        conn.commit()
        conn.close()
    
    def create_user(self, username: str, email: str, password: str, role: str = 'student') -> bool:
        """Create a new user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, role))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user and return user info"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, username, email, password_hash, role FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        conn.close()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user[3]):
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'role': user[4]
            }
        return None
    
    def submit_grievance(self, user_id: int, title: str, category: str, description: str, 
                        sentiment: str, priority: int, image_path: str = None) -> int:
        """Submit a new grievance"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO grievances (user_id, title, category, description, sentiment, priority, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, title, category, description, sentiment, priority, image_path))
        
        grievance_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return grievance_id
    
    def get_grievances(self, user_id: int = None, status: str = None) -> List[Dict]:
        """Get grievances with optional filters"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = '''
            SELECT g.id, g.title, g.category, g.description, g.sentiment, g.priority, 
                   g.status, g.created_at, g.updated_at, g.response, g.rating, g.feedback,
                   u.username, g.user_id, g.image_path
            FROM grievances g
            JOIN users u ON g.user_id = u.id
        '''
        
        conditions = []
        params = []
        
        if user_id:
            conditions.append('g.user_id = ?')
            params.append(user_id)
        
        if status:
            conditions.append('g.status = ?')
            params.append(status)
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        query += ' ORDER BY g.priority ASC, g.created_at DESC'
        
        cursor.execute(query, params)
        grievances = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                'id': g[0], 'title': g[1], 'category': g[2], 'description': g[3],
                'sentiment': g[4], 'priority': g[5], 'status': g[6],
                'created_at': g[7], 'updated_at': g[8], 'response': g[9],
                'rating': g[10], 'feedback': g[11], 'username': g[12], 'user_id': g[13],
                'image_path': g[14]
            }
            for g in grievances
        ]
    
    def update_grievance_status(self, grievance_id: int, status: str, response: str = None):
        """Update grievance status and response"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if status == 'Resolved':
            cursor.execute('''
                UPDATE grievances 
                SET status = ?, response = ?, resolved_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, response, grievance_id))
        else:
            cursor.execute('''
                UPDATE grievances 
                SET status = ?, response = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, response, grievance_id))
        
        conn.commit()
        conn.close()
    
    def add_rating_feedback(self, grievance_id: int, rating: int, feedback: str):
        """Add user rating and feedback for resolved grievance"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE grievances 
            SET rating = ?, feedback = ?
            WHERE id = ?
        ''', (rating, feedback, grievance_id))
        
        conn.commit()
        conn.close()
    
    def get_analytics_data(self) -> Dict[str, Any]:
        """Get analytics data for dashboard"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total grievances
        cursor.execute('SELECT COUNT(*) FROM grievances')
        total_grievances = cursor.fetchone()[0]
        
        # Grievances by status
        cursor.execute('SELECT status, COUNT(*) FROM grievances GROUP BY status')
        status_counts = dict(cursor.fetchall())
        
        # Grievances by category
        cursor.execute('SELECT category, COUNT(*) FROM grievances GROUP BY category')
        category_counts = dict(cursor.fetchall())
        
        # Grievances by sentiment
        cursor.execute('SELECT sentiment, COUNT(*) FROM grievances GROUP BY sentiment')
        sentiment_counts = dict(cursor.fetchall())
        
        # Average rating
        cursor.execute('SELECT AVG(rating) FROM grievances WHERE rating IS NOT NULL')
        avg_rating = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_grievances': total_grievances,
            'status_counts': status_counts,
            'category_counts': category_counts,
            'sentiment_counts': sentiment_counts,
            'avg_rating': round(avg_rating, 2),
            'resolution_times': []  # Placeholder for resolution times
        }
    
    def get_similar_grievances(self, description: str, limit: int = 5) -> List[Dict]:
        """Get similar grievances based on description (simplified implementation)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Simple keyword-based similarity (you can enhance this with more sophisticated NLP)
        keywords = description.lower().split()[:5]  # Use first 5 words as keywords
        
        if not keywords:
            return []
        
        # Create a query to find grievances with similar keywords
        keyword_conditions = []
        params = []
        for keyword in keywords:
            if len(keyword) > 3:  # Only use meaningful words
                keyword_conditions.append("LOWER(g.description) LIKE ?")
                params.append(f"%{keyword}%")
        
        if not keyword_conditions:
            return []
        
        query = f'''
            SELECT DISTINCT g.id, g.title, g.status, g.response
            FROM grievances g
            WHERE ({" OR ".join(keyword_conditions)})
            AND g.status IN ('Resolved', 'In Progress')
            ORDER BY g.created_at DESC
            LIMIT ?
        '''
        params.append(limit)
        
        cursor.execute(query, params)
        similar = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                'id': s[0], 'title': s[1], 'status': s[2], 'response': s[3]
            }
            for s in similar
        ]
    
    def get_unread_notifications_count(self, user_id: int) -> int:
        """Get count of unread notifications for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM notifications WHERE user_id = ? AND is_read = 0', (user_id,))
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def get_notifications(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get notifications for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, grievance_id, message, is_read, created_at, 'Notification' as title
            FROM notifications 
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        notifications = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': n[0], 'grievance_id': n[1], 'message': n[2], 
                'is_read': bool(n[3]), 'created_at': n[4], 'title': n[5]
            }
            for n in notifications
        ]
    
    def mark_notification_as_read(self, notification_id: int):
        """Mark a notification as read"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('UPDATE notifications SET is_read = 1 WHERE id = ?', (notification_id,))
        
        conn.commit()
        conn.close()
    
    def get_recent_activity(self, limit: int = 10) -> List[Dict]:
        """Get recent activity in the system"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = '''
            SELECT u.username, g.title, g.status, g.created_at, g.updated_at
            FROM grievances g
            JOIN users u ON g.user_id = u.id
            ORDER BY g.updated_at DESC
            LIMIT ?
        '''
        
        cursor.execute(query, (limit,))
        activities = cursor.fetchall()
        
        conn.close()
        
        result = []
        for activity in activities:
            username, title, status, created_at, updated_at = activity
            
            # Calculate time ago (simplified)
            from datetime import datetime
            try:
                updated_time = datetime.strptime(updated_at, '%Y-%m-%d %H:%M:%S')
                created_time = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
                now = datetime.now()
                
                if updated_time > created_time:
                    delta = now - updated_time
                    action = f"updated grievance '{title}' to {status}"
                else:
                    delta = now - created_time
                    action = f"submitted grievance '{title}'"
                
                if delta.days > 0:
                    time_ago = f"{delta.days} day(s) ago"
                elif delta.seconds > 3600:
                    time_ago = f"{delta.seconds // 3600} hour(s) ago"
                else:
                    time_ago = f"{delta.seconds // 60} minute(s) ago"
                
                result.append({
                    'username': username,
                    'action': action,
                    'time_ago': time_ago,
                    'details': f"Status: {status}"
                })
            except:
                # Fallback if datetime parsing fails
                result.append({
                    'username': username,
                    'action': f"submitted grievance '{title}'",
                    'time_ago': "recently",
                    'details': f"Status: {status}"
                })
        
        return result
