document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');
    const clearImagesBtn = document.getElementById('clear-images');
    const registerBtn = document.getElementById('register-btn');
    const trainBtn = document.getElementById('train-btn');
    const studentNameInput = document.getElementById('student-name');
    const captureCount = document.getElementById('capture-count');
    const registrationStatus = document.getElementById('registration-status');
    const trainingStatus = document.getElementById('training-status');
    const imageUpload = document.getElementById('image-upload');
    const capturedImagesContainer = document.getElementById('captured-images-container');
    const faceDetectionStatus = document.getElementById('face-detection-status');
    
    // Tab functionality
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Variables to track captured images
    let capturedImages = [];
    let uploadedFiles = [];
    
    // Face detector instance
    let faceDetector = null;
    
    // Initialize camera
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                video.srcObject = stream;
                video.play();
                
                // Initialize face detector once video is playing
                video.onloadedmetadata = function() {
                    // Load face detection script
                    const script = document.createElement('script');
                    script.src = '/static/js/face-detector.js';
                    script.onload = function() {
                        // Initialize face detector
                        faceDetector = new FaceDetector(video, faceDetectionStatus);
                        faceDetector.startDetection();
                    };
                    document.head.appendChild(script);
                };
            })
            .catch(function(error) {
                console.error("Camera error: ", error);
                alert("Could not access the camera. Please check permissions.");
            });
    } else {
        alert("Sorry, your browser does not support camera access.");
    }
    
    // Tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // Remove active class from all buttons and hide all content
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.style.display = 'none');
            
            // Add active class to clicked button and show corresponding content
            this.classList.add('active');
            document.getElementById(`${this.dataset.tab}-tab`).style.display = 'block';
            
            // Start or stop face detection based on active tab
            if (faceDetector) {
                if (this.dataset.tab === 'camera') {
                    faceDetector.startDetection();
                } else {
                    faceDetector.stopDetection();
                }
            }
        });
    });
    
    // Event listeners
    captureBtn.addEventListener('click', captureImage);
    clearImagesBtn.addEventListener('click', clearAllImages);
    registerBtn.addEventListener('click', registerStudent);
    trainBtn.addEventListener('click', trainModel);
    imageUpload.addEventListener('change', handleImageUpload);
    
    // Listen for name input to enable/disable register button
    studentNameInput.addEventListener('input', updateRegisterButtonState);
    
    function updateRegisterButtonState() {
        // Enable register button if we have at least one image and a name
        const hasImages = capturedImages.length > 0 || uploadedFiles.length > 0;
        const hasName = studentNameInput.value.trim() !== '';
        registerBtn.disabled = !(hasImages && hasName);
    }
    
    async function captureImage() {
        // Draw the video frame to the canvas
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // If face detector is available, use it to crop the face
        if (faceDetector) {
            const result = await faceDetector.captureFace(canvas);
            
            if (result.faceDetected) {
                capturedImages.push(result.imageData);
                addImageToPreview(result.imageData);
                updateCaptureUI();
            } else {
                alert('No face detected. Please ensure your face is clearly visible.');
            }
        } else {
            // Fallback to full frame if face detector is not available
            const imageCapture = canvas.toDataURL('image/jpeg');
            capturedImages.push(imageCapture);
            addImageToPreview(imageCapture);
            updateCaptureUI();
        }
    }
    
    function handleImageUpload(e) {
        const files = e.target.files;
        
        if (files.length > 0) {
            // Store the file objects
            uploadedFiles = Array.from(files);
            
            // Preview each image
            for (const file of files) {
                const reader = new FileReader();
                
                reader.onload = function(event) {
                    addImageToPreview(event.target.result);
                    updateCaptureUI();
                };
                
                reader.readAsDataURL(file);
            }
        }
    }
    
    function addImageToPreview(imageSource) {
        // Create container for the image
        const imageContainer = document.createElement('div');
        imageContainer.className = 'image-preview';
        
        // Create the image element
        const img = document.createElement('img');
        img.src = imageSource;
        img.alt = 'Face image';
        
        // Create delete button
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn';
        deleteBtn.innerHTML = '&times;';
        deleteBtn.addEventListener('click', function() {
            // Remove this image from captured images array if it's there
            const index = capturedImages.indexOf(imageSource);
            if (index !== -1) {
                capturedImages.splice(index, 1);
            }
            
            // Remove the container
            imageContainer.remove();
            
            // Update UI
            updateCaptureUI();
        });
        
        // Append elements
        imageContainer.appendChild(img);
        imageContainer.appendChild(deleteBtn);
        capturedImagesContainer.appendChild(imageContainer);
    }
    
    function clearAllImages() {
        // Clear arrays
        capturedImages = [];
        uploadedFiles = [];
        
        // Clear preview container
        capturedImagesContainer.innerHTML = '';
        
        // Reset file input
        imageUpload.value = '';
        
        // Update UI
        updateCaptureUI();
    }
    
    function updateCaptureUI() {
        const totalImages = capturedImages.length + uploadedFiles.length;
        captureCount.textContent = `Capture facial images (${totalImages} captured)`;
        
        // Update register button state
        updateRegisterButtonState();
    }
    
    function registerStudent() {
        if (capturedImages.length === 0 && uploadedFiles.length === 0) {
            alert('Please capture or upload at least one image.');
            return;
        }
        
        if (!studentNameInput.value.trim()) {
            alert('Please enter a student name.');
            return;
        }
        
        // Show loading state
        registrationStatus.textContent = 'Checking for duplicate registration...';
        registrationStatus.className = 'status-message';

        // Create form data for checking duplicate face
        const checkFormData = new FormData();
        if (capturedImages.length > 0) {
            checkFormData.append('image', capturedImages[0]);
        } else if (uploadedFiles.length > 0) {
            // For uploaded files, we'll need to send the first file
            checkFormData.append('uploaded_image', uploadedFiles[0]);
        }

        // First check for duplicate face
        fetch('/api/check_duplicate_face', {
            method: 'POST',
            body: checkFormData
        })
        .then(response => response.json())
        .then(data => {
            if (data.exists) {
                // Handle duplicate face
                const studentName = data.student_name || 'Unknown Student';
                registrationStatus.textContent = `This person is already registered as "${studentName}" (ID: ${data.student_id})`;
                registrationStatus.className = 'status-message error';
                return Promise.reject(new Error('Duplicate registration'));
            } else if (data.message) {
                // Show informational message but continue registration
                registrationStatus.textContent = data.message;
                registrationStatus.className = 'status-message info';
            }

            // Continue with registration if no duplicate
            registrationStatus.textContent = 'Registering student...';
            
            // Create form data for registration
            const formData = new FormData();
            formData.append('name', studentNameInput.value);
            formData.append('image_count', capturedImages.length);
            
            // Add captured images
            for (let i = 0; i < capturedImages.length; i++) {
                formData.append(`image_${i}`, capturedImages[i]);
            }
            
            // Add uploaded files
            for (const file of uploadedFiles) {
                formData.append('uploaded_images', file);
            }
            
            // Send to server
            return fetch('/api/register_student', {
                method: 'POST',
                body: formData
            });
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                registrationStatus.textContent = `Student registered successfully with ID: ${data.student_id}`;
                registrationStatus.className = 'status-message success';
                
                // Reset form
                studentNameInput.value = '';
                clearAllImages();
            } else {
                registrationStatus.textContent = 'Failed to register student.';
                registrationStatus.className = 'status-message error';
            }
        })
        .catch(error => {
            if (error.message !== 'Duplicate registration') {
                console.error('Error:', error);
                registrationStatus.textContent = 'Error registering student.';
                registrationStatus.className = 'status-message error';
            }
        });
    }
    
    function trainModel() {
        // Ask user if they want to force a full retraining
        const forceRetrain = confirm(
            "Do you want to perform a full retraining of all students?\n\n" +
            "NO = Train only new students that haven't been processed yet (faster)\n" +
            "YES = Retrain all students from scratch (slower but may be more accurate)"
        );
        
        // Show loading state
        trainingStatus.textContent = forceRetrain ? 
            'Training full model... This may take a few minutes.' : 
            'Training model for new students... This may take a moment.';
        trainingStatus.className = 'status-message';
        
        // Send training request with force_retrain parameter
        fetch('/api/train_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                force_retrain: forceRetrain
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const message = data.full_retrain ? 
                    `Model fully retrained with ${data.encoding_count} encodings!` : 
                    `Model updated with ${data.encoding_count} total encodings!`;
                
                trainingStatus.textContent = message;
                trainingStatus.className = 'status-message success';
            } else {
                trainingStatus.textContent = 'Failed to train model.';
                trainingStatus.className = 'status-message error';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            trainingStatus.textContent = 'Error training model.';
            trainingStatus.className = 'status-message error';
        });
    }
    
    // Initialize UI
    updateCaptureUI();
});