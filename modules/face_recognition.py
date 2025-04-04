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
        import concurrent.futures
        import time
        
        start_time = time.time()
        print("Starting face recognition model training...")
        
        encodings = []
        names = []
        
        # Get all student directories
        student_dirs = os.listdir('data/student_images')
        total_students = len(student_dirs)
        processed_students = 0
        
        # Function to process a single student's images
        def process_student(student_id):
            student_encodings = []
            student_dir = f'data/student_images/{student_id}'
            if not os.path.isdir(student_dir):
                return [], []
                
            # Get image files and limit to a reasonable number (e.g., 20 max per student)
            image_files = [f for f in os.listdir(student_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            max_images_per_student = 20  # Limiting images per student for faster training
            if len(image_files) > max_images_per_student:
                # Use a subset with even distribution
                image_files = image_files[::len(image_files) // max_images_per_student][:max_images_per_student]
            
            # Return early if no images
            if not image_files:
                return [], []
                
            # Process each image for this student
            for img_file in image_files:
                img_path = f'{student_dir}/{img_file}'
                try:
                    # Load image and find face encodings - use lower resolution for speed
                    image = face_recognition.load_image_file(img_path)
                    
                    # Resize large images for better performance
                    h, w = image.shape[:2]
                    if max(h, w) > 640:  # If image is larger than 640px in any dimension
                        scale = 640 / max(h, w)
                        image = cv2.resize(image, (int(w * scale), int(h * scale)))
                    
                    # Use lower jitter values for faster training (1 instead of 3)
                    face_encodings = face_recognition.face_encodings(image, num_jitters=1)
                    
                    # If a face was found, add it to our training data
                    if face_encodings:
                        student_encodings.append(face_encodings[0])
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    
            # Only add one encoding per student if we're in a hurry
            # (uncomment this if you want faster training but less accuracy)
            # if student_encodings:
            #     return [student_encodings[0]], [student_id]
            
            # Return all encodings for this student
            return student_encodings, [student_id] * len(student_encodings)
        
        # Process students in parallel (but limit workers to avoid excessive memory usage)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_student, student_id) for student_id in student_dirs]
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    student_encodings, student_ids = future.result()
                    encodings.extend(student_encodings)
                    names.extend(student_ids)
                    processed_students += 1
                    
                    # Print progress update
                    print(f"Processed {processed_students}/{total_students} students ({len(encodings)} face encodings)")
        except Exception as e:
            print(f"Error in parallel processing: {e}")
            # Fall back to sequential processing if parallel fails
            for student_id in student_dirs:
                student_encodings, student_ids = process_student(student_id)
                encodings.extend(student_encodings) 
                names.extend(student_ids)
        
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
        
        # Calculate training time
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds with {len(encodings)} face encodings")
        
        return len(encodings)
    
    def recognize_faces(self, image):
        """Recognize faces in the given image"""
        # If model is not trained, return empty list
        if not self.known_face_encodings:
            return []
        
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return []
        
        # Convert to RGB if needed (face_recognition uses RGB)
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = processed_image
            
        # Find faces - try with multiple detection parameters
        face_locations = face_recognition.face_locations(
            rgb_image, 
            model='hog',  # Use 'cnn' for more accuracy if GPU available
            number_of_times_to_upsample=2  # Increase to detect smaller faces
        )
        
        # If no faces found, try again with different parameters
        if not face_locations:
            face_locations = face_recognition.face_locations(
                rgb_image, 
                model='hog',
                number_of_times_to_upsample=1
            )
        
        if not face_locations:
            return []
            
        # Get encodings with enhanced parameters
        face_encodings = face_recognition.face_encodings(
            rgb_image, 
            face_locations,
            num_jitters=2  # Increase for more accurate encoding
        )
        
        recognized_students = []
        
        # Compare with known faces with stricter threshold
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding,
                tolerance=0.5  # Lower tolerance for stricter matching
            )
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                # Use a stricter threshold for more accurate matching
                if matches[best_match_index] and face_distances[best_match_index] < 0.5:
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
            
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return False, None
            
        # Convert to RGB if needed
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = processed_image
        
        # Find faces in the image with enhanced parameters
        face_locations = face_recognition.face_locations(
            rgb_image, 
            model='hog',
            number_of_times_to_upsample=2
        )
        
        if not face_locations:
            return False, None  # No faces detected
            
        # Get encodings with higher accuracy settings
        face_encodings = face_recognition.face_encodings(
            rgb_image, 
            face_locations,
            num_jitters=3
        )
            
        if not face_encodings:
            return False, None  # Could not extract features from face
            
        # Get the first face in the image
        face_encoding = face_encodings[0]
        
        # Compare with known faces with stricter threshold
        matches = face_recognition.compare_faces(
            self.known_face_encodings, 
            face_encoding,
            tolerance=0.5
        )
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < 0.5:  # Stricter threshold
                try:
                    # Convert to int to ensure it's a valid ID
                    student_id = int(self.known_face_names[best_match_index])
                    return True, student_id
                except (ValueError, TypeError):
                    # If ID is not valid, return no match
                    print(f"Warning: Invalid student ID in face recognition model: {self.known_face_names[best_match_index]}")
                    return False, None
                
        return False, None
    
    def preprocess_image(self, image):
        """Preprocess image for better face detection"""
        # Check if image is valid
        if image is None or image.size == 0:
            return None
            
        # Make a copy to avoid modifying the original
        processed_image = image.copy()
        
        # Resize if too large (helps with performance)
        max_size = 1024
        height, width = processed_image.shape[:2]
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            processed_image = cv2.resize(processed_image, (int(width * scale), int(height * scale)))
        
        # Convert BGR to RGB if needed
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            # Check if we need to convert from BGR to RGB
            if cv2.COLOR_BGR2RGB:
                rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = processed_image
        else:
            # If grayscale, convert to RGB
            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        
        # Apply image enhancements for better face detection
        try:
            # Improve contrast using CLAHE
            lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge back the channels
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Convert back to BGR for OpenCV
            return enhanced_image
        except:
            # If enhancement fails, return the original RGB image
            return rgb_image

    def detect_and_crop_face(self, image, padding=0.2):
        """
        Detect face in image and crop to just the face with some padding
        
        Args:
            image: OpenCV image (numpy array) in BGR format
            padding: Percentage of padding to add around face (0.2 = 20%)
            
        Returns:
            Cropped image containing just the face, or None if no face detected
        """
        # Convert to RGB for face_recognition library
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face locations - upsampling helps detect smaller faces
        face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=1)
        
        if not face_locations:
            return None  # No face detected
        
        # Use the first face if multiple are detected
        top, right, bottom, left = face_locations[0]
        
        # Calculate padding
        height = bottom - top
        width = right - left
        padding_h = int(height * padding)
        padding_w = int(width * padding)
        
        # Add padding with boundary checks
        h, w = image.shape[:2]
        top = max(0, top - padding_h)
        left = max(0, left - padding_w)
        bottom = min(h, bottom + padding_h)
        right = min(w, right + padding_w)
        
        # Crop image to face region
        face_image = image[top:bottom, left:right]
        
        return face_image

    def extract_all_faces(self, image, padding=0.2):
        """
        Extract all faces from an image with padding
        
        Args:
            image: OpenCV image (numpy array)
            padding: Percentage of padding to add around faces
            
        Returns:
            List of cropped face images
        """
        # Convert to RGB for face_recognition library
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=1)
        
        faces = []
        h, w = image.shape[:2]
        
        for face_location in face_locations:
            top, right, bottom, left = face_location
            
            # Calculate padding
            height = bottom - top
            width = right - left
            padding_h = int(height * padding)
            padding_w = int(width * padding)
            
            # Add padding with boundary checks
            top = max(0, top - padding_h)
            left = max(0, left - padding_w)
            bottom = min(h, bottom + padding_h)
            right = min(w, right + padding_w)
            
            # Crop image to face region
            face_image = image[top:bottom, left:right]
            faces.append(face_image)
        
        return faces