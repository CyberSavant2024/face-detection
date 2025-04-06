# Smart Attendance System

A face recognition-based attendance tracking system with a Flask backend and React frontend.

## Overview

The Smart Attendance System automates attendance tracking using facial recognition technology. It allows administrators to register students by capturing their facial images and then mark attendance by recognizing those students in subsequently captured images.

## Features

- **Student Registration**: Register students by capturing multiple face images or uploading images
- **Face Recognition**: Accurately identify registered students in images
- **Attendance Tracking**: Mark and track attendance with dates and timestamps
- **Duplicate Prevention**: Prevents the same person from being registered twice
- **Responsive UI**: Works on both desktop and mobile devices

## Technologies Used

### Backend
- Python 3.x
- Flask (Web Framework)
- OpenCV and face_recognition library for facial recognition
- SQLite for database storage

### Frontend
- React.js
- React Router for navigation
- Axios for API communication
- Custom CSS for styling

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm 6 or higher
- Required Python packages: flask, flask-cors, opencv-python, face-recognition, numpy

### Backend Setup

1. Clone the repository:
   ```
   git clone <your-repository-url>
   cd smart-attendance-system
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```
   pip install flask flask-cors opencv-python face-recognition numpy
   ```

4. Run the Flask server:
   ```
   python app.py
   ```
   The server will start on http://localhost:5000

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```
   The React app will start on http://localhost:3000

## Usage

### Registering Students

1. Go to the "Register Student" page
2. Enter the student's name
3. Capture multiple face images of the student using the webcam or upload existing images
4. Click "Register Student" to save the student's information
5. Click "Train Model" to update the face recognition model

### Taking Attendance

1. Go to the "Take Attendance" page
2. Select the date for attendance
3. Capture an image of the student(s) or upload an image
4. Click "Process Attendance" to mark attendance for recognized students
5. View the attendance records for the selected date at the bottom of the page

### Administration

- Use the "Reset System" button on the home page to clear all student data and attendance records (use with caution)

## Project Structure

- `/app.py` - Main Flask application
- `/modules` - Python modules for face recognition and database operations
- `/data` - Directory for storing student images and trained models
  - `/data/student_images` - Student face images organized by ID
  - `/data/models` - Trained face recognition models
- `/static` - Static files for the HTML/CSS/JS version
- `/templates` - HTML templates for the non-React version
- `/frontend` - React frontend application
  - `/frontend/src/components` - React components
  - `/frontend/src/services` - API service functions






