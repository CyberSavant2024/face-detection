from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_cors import CORS  # Add this line
import os
import json
import datetime
import cv2
import numpy as np
import shutil
from modules.face_recognition import FaceRecognizer
from modules.database import Database

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

face_recognizer = FaceRecognizer()
db = Database('attendance_db.sqlite')

# Ensure required directories exist
os.makedirs('data/student_images', exist_ok=True)
os.makedirs('data/models', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/api/students', methods=['GET'])
def get_students():
    students = db.get_all_students()
    return jsonify(students)

@app.route('/api/register_student', methods=['POST'])
def register_student():
    name = request.form.get('name')
    student_id = db.add_student(name)
    
    # Save images - now supporting many images from continuous capture
    images = []
    image_count = int(request.form.get('image_count', 0))
    
    save_path = f'data/student_images/{student_id}'
    os.makedirs(save_path, exist_ok=True)
    
    # Process in chunks to handle potentially large numbers of images
    batch_size = 10  # Process 10 images at a time
    for batch_start in range(0, image_count, batch_size):
        batch_end = min(batch_start + batch_size, image_count)
        
        for i in range(batch_start, batch_end):
            image_data = request.form.get(f'image_{i}')
            if image_data:
                try:
                    # Convert base64 to image and save
                    image = face_recognizer.base64_to_image(image_data)
                    
                    # Already cropped on client side, but double-check for better face detection
                    face_image = face_recognizer.detect_and_crop_face(image)
                    if face_image is not None:
                        cv2.imwrite(f'{save_path}/{i}.jpg', face_image)
                        images.append(face_image)
                    else:
                        # Fall back to the original image if face detection fails
                        cv2.imwrite(f'{save_path}/{i}.jpg', image)
                        images.append(image)
                        print(f"Warning: No face detected in image {i}, using full image")
                except Exception as e:
                    print(f"Error processing image {i}: {e}")
    
    # Handle uploaded files
    uploaded_files = request.files.getlist('uploaded_images')
    for i, file in enumerate(uploaded_files):
        if file.filename:
            try:
                temp_path = f'temp_upload_{i}.jpg'
                file.save(temp_path)
                
                # Read the uploaded image
                image = cv2.imread(temp_path)
                
                # Detect and crop face
                face_image = face_recognizer.detect_and_crop_face(image)
                
                if face_image is not None:
                    # Save the cropped face
                    cv2.imwrite(f'{save_path}/{image_count + i}.jpg', face_image)
                else:
                    # Save the original image if no face detected
                    cv2.imwrite(f'{save_path}/{image_count + i}.jpg', image)
                    print(f"Warning: No face detected in uploaded file {i}, using full image")
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                print(f"Error processing uploaded file {i}: {e}")
    
    print(f"Registered student {name} with ID {student_id} and {len(images)} images")
    
    # Train only this new student instead of rebuilding the entire model
    encoding_count = face_recognizer.train_student(student_id)
    
    return jsonify({
        'success': True, 
        'student_id': student_id,
        'encoding_count': encoding_count
    })

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Train the face recognition model - either incrementally or full retraining"""
    # Check if we're forcing a full retrain
    force_retrain = request.json.get('force_retrain', False) if request.is_json else False
    
    encoding_count = face_recognizer.train_model(force_retrain=force_retrain)
    
    return jsonify({
        'success': True, 
        'encoding_count': encoding_count, 
        'full_retrain': force_retrain
    })

@app.route('/api/take_attendance', methods=['POST'])
def take_attendance():
    date = request.form.get('date')
    
    # Handle both captured and uploaded images
    recognized_students = []
    
    # Check if captured image is provided
    image_data = request.form.get('image')
    if image_data:
        image = face_recognizer.base64_to_image(image_data)
        
        # For attendance, we want to detect all faces in the image
        face_images = face_recognizer.extract_all_faces(image)
        
        if face_images:
            # Process each detected face
            for face_img in face_images:
                students = face_recognizer.recognize_faces(face_img)
                recognized_students.extend(students)
        else:
            # Fall back to the full image if no faces detected
            students = face_recognizer.recognize_faces(image)
            recognized_students.extend(students)
    
    # Handle uploaded image similarly
    if 'uploaded_image' in request.files:
        uploaded_file = request.files['uploaded_image']
        if uploaded_file.filename:
            # Save temporarily
            temp_path = 'temp_attendance.jpg'
            uploaded_file.save(temp_path)
            
            # Process the image
            image = cv2.imread(temp_path)
            if image is not None:
                # Extract all faces
                face_images = face_recognizer.extract_all_faces(image)
                
                if face_images:
                    # Process each detected face
                    for face_img in face_images:
                        students = face_recognizer.recognize_faces(face_img)
                        recognized_students.extend(students)
                else:
                    # Fall back to the full image
                    students = face_recognizer.recognize_faces(image)
                    recognized_students.extend(students)
            
            # Remove temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Remove duplicates
    recognized_students = list(set(recognized_students))
    
    # Record attendance
    for student_id in recognized_students:
        db.mark_attendance(student_id, date)
    
    return jsonify({'success': True, 'recognized': recognized_students})

@app.route('/api/attendance_data', methods=['GET'])
def get_attendance_data():
    date = request.args.get('date', datetime.date.today().isoformat())
    attendance_data = db.get_attendance_by_date(date)
    
    # Get student details for recognized students
    attendance_with_details = []
    for record in attendance_data:
        student_details = db.get_student_by_id(record['student_id'])
        combined = {**record, **student_details}
        attendance_with_details.append(combined)
    
    return jsonify(attendance_with_details)

@app.route('/api/reset_system', methods=['POST'])
def reset_system():
    try:
        # Drop and recreate database
        db.reset_database()
        
        # Remove all student images
        if os.path.exists('data/student_images'):
            shutil.rmtree('data/student_images')
            os.makedirs('data/student_images')
        
        # Remove trained models
        if os.path.exists('data/models'):
            shutil.rmtree('data/models')
            os.makedirs('data/models')
        
        return jsonify({'success': True, 'message': 'System reset successful'})
    except Exception as e:
        print(f"Error during reset: {e}")
        return jsonify({'success': False, 'message': f'Error during reset: {str(e)}'})

@app.route('/data/student_images/<path:filename>')
def student_images(filename):
    return send_from_directory('data/student_images', filename)

@app.route('/api/check_duplicate_face', methods=['POST'])
def check_duplicate_face():
    try:
        # Get image from request
        image_data = request.form.get('image')
        
        if not image_data:
            return jsonify({
                'success': False,
                'message': 'No image provided'
            })
        
        # Convert base64 to image
        image = face_recognizer.base64_to_image(image_data)
        
        # Check if face exists
        exists, student_id = face_recognizer.check_face_exists(image)
        
        if exists and student_id:
            # Get student details
            student = db.get_student_by_id(student_id)
            
            # Check if we actually got a valid student name
            student_name = student.get('name')
            if not student_name or student_name == 'Unknown Student':
                # If the student record doesn't exist anymore
                print(f"Warning: Face recognized with ID {student_id} but no matching student record found")
                
                # Retrain the model to remove this face encoding
                face_recognizer.train_model()
                
                return jsonify({
                    'success': True,
                    'exists': False,
                    'message': 'Face data was outdated and has been updated. Please try again.'
                })
            
            return jsonify({
                'success': True,
                'exists': True,
                'student_id': student_id,
                'student_name': student_name
            })
        else:
            return jsonify({
                'success': True,
                'exists': False
            })
    except Exception as e:
        print(f"Error checking duplicate face: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

# Add this route to serve the React app in production
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join('frontend/build', path)):
        return send_from_directory('frontend/build', path)
    return send_from_directory('frontend/build', 'index.html')

if __name__ == '__main__':
    db.setup_database()  # Ensure database is set up
    app.run(debug=True, host='0.0.0.0', port=5000)