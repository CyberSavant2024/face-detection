import sqlite3
import datetime
import os

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
    
    def get_connection(self):
        """Get a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def setup_database(self):
        """Create database tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                registration_date TEXT NOT NULL
            )
        ''')
        
        # Create attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students (id),
                UNIQUE(student_id, date)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def reset_database(self):
        """Completely reset the database by dropping all tables and recreating them"""
        # Close any open connections
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Drop all tables
            cursor.execute("DROP TABLE IF EXISTS attendance")
            cursor.execute("DROP TABLE IF EXISTS students")
            
            conn.commit()
            conn.close()
            
            # Recreate the database structure
            self.setup_database()
            
            return True
        except Exception as e:
            print(f"Error resetting database: {e}")
            return False
    
    def add_student(self, name):
        """Add a new student and return their ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        today = datetime.date.today().isoformat()
        cursor.execute(
            "INSERT INTO students (name, registration_date) VALUES (?, ?)",
            (name, today)
        )
        
        student_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return student_id
    
    def get_student_by_id(self, student_id):
        """Get student details by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, registration_date FROM students WHERE id = ?", (student_id,))
        student = cursor.fetchone()
        
        # Properly handle the case where no student record is found
        if student:
            result = dict(student)
        else:
            result = {'id': student_id, 'name': 'Unknown Student', 'registration_date': None}
        
        conn.close()
        return result
    
    def mark_attendance(self, student_id, date):
        """Mark attendance for a student on a given date"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        now = datetime.datetime.now().isoformat()
        
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO attendance (student_id, date, timestamp) VALUES (?, ?, ?)",
                (student_id, date, now)
            )
            conn.commit()
            success = True
        except Exception as e:
            print(f"Error marking attendance: {e}")
            success = False
        
        conn.close()
        return success
    
    def get_all_students(self):
        """Get all registered students"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, registration_date FROM students")
        students = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return students
    
    def get_attendance_by_date(self, date):
        """Get attendance records for a specific date"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT a.id, a.student_id, s.name, a.timestamp 
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date = ?
        ''', (date,))
        
        attendance = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return attendance