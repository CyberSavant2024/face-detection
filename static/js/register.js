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
    
    // Tab functionality
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Variables to track captured images
    let capturedImages = [];
    let uploadedFiles = [];
    
    // Initialize camera
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                video.srcObject = stream;
                video.play();
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
    
    function captureImage() {
        const context = canvas.getContext('2d');
        // Draw the video frame to the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64
        const imageCapture = canvas.toDataURL('image/jpeg');
        capturedImages.push(imageCapture);
        
        // Add to preview container
        addImageToPreview(imageCapture);
        
        // Update UI
        updateCaptureUI();
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
        registrationStatus.textContent = 'Registering student...';
        registrationStatus.className = 'status-message';
        
        // Create form data
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
        fetch('/api/register_student', {
            method: 'POST',
            body: formData
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
            console.error('Error:', error);
            registrationStatus.textContent = 'Error registering student.';
            registrationStatus.className = 'status-message error';
        });
    }
    
    function trainModel() {
        // Show loading state
        trainingStatus.textContent = 'Training model... This may take a moment.';
        trainingStatus.className = 'status-message';
        
        // Send training request
        fetch('/api/train_model', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                trainingStatus.textContent = 'Model trained successfully!';
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