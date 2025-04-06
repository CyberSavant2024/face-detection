import os
import cv2
import numpy as np
import pickle
import face_recognition
import base64
from io import BytesIO
import time
import dlib  # Add dlib import
import json

class FaceRecognizer:
    def __init__(self, model_path='data/models/face_model.pkl', metadata_path='data/models/trained_students.json'):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.trained_students = set()  # Keep track of trained student IDs
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        self.load_model()
        self.load_trained_students()
        
        # Check for GPU availability
        self.use_gpu = dlib.DLIB_USE_CUDA and dlib.cuda.get_num_devices() > 0
        if self.use_gpu:
            print(f"GPU acceleration available. Using {dlib.cuda.get_num_devices()} CUDA device(s)")
        else:
            print("GPU acceleration not available. Using CPU only")
    
    def load_model(self):
        """Load the trained model if it exists"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
                print(f"Loaded model with {len(self.known_face_encodings)} face encodings")
    
    def load_trained_students(self):
        """Load the set of already trained student IDs"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    # Convert student IDs to strings for consistency
                    self.trained_students = set(str(id) for id in metadata.get('trained_students', []))
                print(f"Loaded metadata for {len(self.trained_students)} trained students")
            except Exception as e:
                print(f"Error loading trained students metadata: {e}")
                self.trained_students = set()
        else:
            # Initialize from known_face_names if metadata doesn't exist
            self.trained_students = set(str(id) for id in set(self.known_face_names))
            self.save_trained_students()
    
    def save_trained_students(self):
        """Save the set of trained student IDs to persistent storage"""
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w') as f:
            json.dump({
                'trained_students': list(self.trained_students),
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f)
    
    def save_model(self):
        """Save the current model state"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)
        print(f"Model saved with {len(self.known_face_encodings)} face encodings")
    
    def train_student(self, student_id):
        """Train model for a single student and update the main model
        
        Args:
            student_id: ID of the student to train
            
        Returns:
            int: Number of face encodings extracted for this student
        """
        # Convert to string for consistent comparison
        student_id = str(student_id)
        
        # Skip if student is already trained
        if student_id in self.trained_students:
            print(f"Student {student_id} is already trained. Skipping.")
            return 0
            
        print(f"Training model for student {student_id}...")
        start_time = time.time()
        
        student_dir = f'data/student_images/{student_id}'
        if not os.path.isdir(student_dir):
            print(f"No directory found for student {student_id}")
            return 0
        
        # Get image files
        image_files = [f for f in os.listdir(student_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"No images found for student {student_id}")
            return 0
        
        # Limit images per student for faster training
        max_images_per_student = 20
        
        if len(image_files) > max_images_per_student:
            # Use a subset with even distribution
            image_files = image_files[::len(image_files) // max_images_per_student][:max_images_per_student]
        
        # Process all images for this student
        student_encodings = []
        
        for img_file in image_files:
            img_path = f'{student_dir}/{img_file}'
            try:
                # Load image
                image = cv2.imread(img_path)
                
                # Skip invalid images
                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                    
                # Convert BGR to RGB (face_recognition uses RGB)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Use HOG face detector which is CPU-friendly
                face_locations = face_recognition.face_locations(
                    rgb_image, 
                    model='hog',
                    number_of_times_to_upsample=1
                )
                
                # If no face found, try again with higher upsample
                if not face_locations:
                    face_locations = face_recognition.face_locations(
                        rgb_image,
                        model='hog',
                        number_of_times_to_upsample=2
                    )
                
                # Keep jitters low for CPU
                num_jitters = 1
                
                # Get face encodings
                face_encodings = face_recognition.face_encodings(
                    rgb_image,
                    face_locations,
                    num_jitters=num_jitters,
                    model="small"  # Use small model for faster processing
                )
                
                # If a face was found, add it to our training data
                if face_encodings:
                    student_encodings.append(face_encodings[0])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # If we have encodings for this student
        if student_encodings:
            # Add to the main model
            self.known_face_encodings.extend(student_encodings)
            self.known_face_names.extend([student_id] * len(student_encodings))
            
            # Mark the student as trained
            self.trained_students.add(student_id)
            
            # Save the updated model and trained students list
            self.save_model()
            self.save_trained_students()
            
            total_time = time.time() - start_time
            print(f"Added {len(student_encodings)} encodings for student {student_id} in {total_time:.2f} seconds")
            return len(student_encodings)
        
        return 0
    
    def train_model(self, force_retrain=False):
        """Train facial recognition model using saved student images
        
        Args:
            force_retrain: If True, reprocess all students even if already trained
        
        Returns:
            int: Total number of face encodings in the model
        """
        import concurrent.futures
        
        start_time = time.time()
        print("Starting face recognition model training...")
        
        # If force retrain, clear existing data
        if force_retrain:
            print("Forcing full retraining of model (processing all students)")
            encodings = []
            names = []
            self.trained_students = set()  # Reset the trained students set
        else:
            # Keep existing encodings and only add new students
            encodings = self.known_face_encodings.copy()
            names = self.known_face_names.copy()
            print(f"Incremental training: only processing new students. Currently have {len(encodings)} encodings.")
        
        # Get all student directories
        try:
            student_dirs = os.listdir('data/student_images')
        except FileNotFoundError:
            os.makedirs('data/student_images', exist_ok=True)
            student_dirs = []
            
        total_students = len(student_dirs)
        processed_students = 0
        new_encodings = 0
        
        # Function to process a single student's images (reusing existing code)
        def process_student(student_id):
            # Skip already trained students unless force_retrain is True
            if student_id in self.trained_students and not force_retrain:
                return [], []
                
            student_encodings = []
            student_dir = f'data/student_images/{student_id}'
            if not os.path.isdir(student_dir):
                return [], []
                
            # Get image files
            image_files = [f for f in os.listdir(student_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # ... existing code for processing images ...
            
            # Return all encodings for this student
            return student_encodings, [student_id] * len(student_encodings)
        
        try:
            # Process only untrained students (or all if force_retrain)
            students_to_process = []
            for student_id in student_dirs:
                if student_id not in self.trained_students or force_retrain:
                    students_to_process.append(student_id)
            
            if not students_to_process:
                print("No new students to train!")
                return len(encodings)
            
            print(f"Processing {len(students_to_process)} students out of {total_students} total")
            
            # Process students in parallel with CPU-friendly settings
            max_workers = 4  # Good balance for most systems
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_student, student_id) for student_id in students_to_process]
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    student_encodings, student_ids = future.result()
                    encodings.extend(student_encodings)
                    names.extend(student_ids)
                    new_encodings += len(student_encodings)
                    processed_students += 1
                    
                    # Add to trained students if we got encodings
                    if student_encodings and student_ids:
                        self.trained_students.add(str(student_ids[0]))
                    
                    # Print progress update
                    print(f"Processed {processed_students}/{len(students_to_process)} students ({new_encodings} new face encodings)")
        
        except Exception as e:
            print(f"Error in parallel processing: {e}")
            # Fall back to sequential processing
            for student_id in students_to_process:
                student_encodings, student_ids = process_student(student_id)
                encodings.extend(student_encodings)
                names.extend(student_ids)
                if student_encodings and student_ids:
                    self.trained_students.add(str(student_ids[0]))
        
        # Save the trained model
        self.known_face_encodings = encodings
        self.known_face_names = names
        self.save_model()
        self.save_trained_students()
        
        # Calculate training time
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds with {len(encodings)} total face encodings")
        print(f"Added {new_encodings} new encodings to the model")
        
        return len(encodings)
    
    # Modified to use GPU acceleration when available
    def recognize_faces(self, image):
        """Recognize faces in the given image with GPU acceleration if available"""
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
            
        # Find faces with GPU-optimized model if available
        face_locations = face_recognition.face_locations(
            rgb_image, 
            model='cnn' if self.use_gpu else 'hog',
            number_of_times_to_upsample=1
        )
        
        # If no faces found, try again with different parameters
        if not face_locations:
            face_locations = face_recognition.face_locations(
                rgb_image, 
                model='hog',  # Fall back to HOG
                number_of_times_to_upsample=2
            )
        
        if not face_locations:
            return []
            
        # Get encodings with enhanced parameters and GPU acceleration
        face_encodings = face_recognition.face_encodings(
            rgb_image, 
            face_locations,
            num_jitters=5 if self.use_gpu else 1,  # More jitters if GPU available
            model="large" if self.use_gpu else "small"  # Use more accurate model with GPU
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