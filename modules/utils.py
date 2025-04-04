import os
import datetime

def get_today_date():
    """Get today's date in ISO format (YYYY-MM-DD)"""
    return datetime.date.today().isoformat()

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        'data/student_images',
        'data/models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)