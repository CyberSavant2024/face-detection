import os
import cv2
import numpy as np
import pickle
import face_recognition
import base64
from io import BytesIO

class FaceRecognizer:
    def __init__(self, model_path='data/models/face_model.pkl'):
        self.model_path = model_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_model()
    
    def load_model(self):
        """Load the trained model if it exists"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
    
    def train_model(self):
        """Train facial recognition model using saved student images"""
        encodings = []
        names = []
        
        # Get all student directories
        student_dirs = os.listdir('data/student_images')
        
        for student_id in student_dirs:
            student_dir = f'data/student_images/{student_id}'
            if os.path.isdir(student_dir):
                # Process each image for this student
                for img_file in os.listdir(student_dir):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = f'{student_dir}/{img_file}'
                        # Load image and find face encodings
                        image = face_recognition.load_image_file(img_path)
                        face_encodings = face_recognition.face_encodings(image)
                        
                        # If a face was found, add it to our training data
                        if face_encodings:
                            encodings.append(face_encodings[0])
                            names.append(student_id)
        
        # Save the trained model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'encodings': encodings,
                'names': names
            }, f)
        
        # Update the current model
        self.known_face_encodings = encodings
        self.known_face_names = names
        
        return len(encodings)
    
    def recognize_faces(self, image):
        """Recognize faces in the given image"""
        # If model is not trained, return empty list
        if not self.known_face_encodings:
            return []
        
        # Find faces in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        recognized_students = []
        
        # Compare with known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            student_id = "unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                student_id = self.known_face_names[best_match_index]
                
                # Only add unique student IDs
                if student_id not in recognized_students:
                    recognized_students.append(student_id)
        
        return recognized_students
    
    def base64_to_image(self, base64_string):
        """Convert base64 string to an OpenCV image"""
        # Remove the data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
            
        # Decode base64 string
        img_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img

    def check_face_exists(self, image):
        """Check if the face in the image exists in the database
        
        Returns:
            tuple: (exists, student_id) where:
                - exists is a boolean indicating if the face exists
                - student_id is the ID of the matching student if exists is True, None otherwise
        """
        # If model is not trained, no faces exist yet
        if not self.known_face_encodings:
            return False, None
        
        # Find faces in the image
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            return False, None  # No faces detected
            
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            return False, None  # Could not extract features from face
            
        # Get the first face in the image
        face_encoding = face_encodings[0]
        
        # Compare with known faces
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < 0.6:  # Threshold for face matching
                student_id = self.known_face_names[best_match_index]
                return True, student_id
        
        return False, None