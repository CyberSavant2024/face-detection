document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');
    const retakeBtn = document.getElementById('retake-btn');
    const previewContainer = document.getElementById('preview-container');
    const capturedImage = document.getElementById('captured-image');
    const attendanceList = document.getElementById('attendance-list').querySelector('tbody');
    const dateInput = document.getElementById('attendance-date');
    const selectedDateSpan = document.getElementById('selected-date');
    const recognitionResult = document.getElementById('recognition-result');
    
    // Set default date to today
    const today = new Date().toISOString().split('T')[0];
    dateInput.value = today;
    selectedDateSpan.textContent = formatDate(today);
    
    // Variables to store capture
    let imageCapture = null;
    
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
    
    // Load initial attendance data
    loadAttendanceData(dateInput.value);
    
    // Event listeners
    captureBtn.addEventListener('click', captureImage);
    retakeBtn.addEventListener('click', retakeImage);
    dateInput.addEventListener('change', function() {
        selectedDateSpan.textContent = formatDate(this.value);
        loadAttendanceData(this.value);
    });
    
    function captureImage() {
        const context = canvas.getContext('2d');
        // Draw the video frame to the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64
        imageCapture = canvas.toDataURL('image/jpeg');
        
        // Show the captured image
        capturedImage.src = imageCapture;
        previewContainer.style.display = 'block';
        
        // Change buttons
        captureBtn.style.display = 'none';
        retakeBtn.style.display = 'inline-block';
        
        // Process the image for attendance
        processAttendance(imageCapture);
    }
    
    function retakeImage() {
        // Clear the image
        imageCapture = null;
        previewContainer.style.display = 'none';
        
        // Reset buttons
        captureBtn.style.display = 'inline-block';
        retakeBtn.style.display = 'none';
        
        // Reset recognition result
        recognitionResult.innerHTML = '<p>Capture an image to mark attendance.</p>';
    }
    
    function processAttendance(imageData) {
        // Show loading state
        recognitionResult.innerHTML = '<p>Processing attendance...</p>';
        
        // Create form data
        const formData = new FormData();
        formData.append('date', dateInput.value);
        formData.append('image', imageData);
        
        // Send to server
        fetch('/api/take_attendance', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (data.recognized && data.recognized.length > 0) {
                    recognitionResult.innerHTML = `<p class="success">Attendance marked for ${data.recognized.length} student(s).</p>`;
                    // Reload attendance data
                    loadAttendanceData(dateInput.value);
                } else {
                    recognitionResult.innerHTML = '<p>No registered students recognized in the image.</p>';
                }
            } else {
                recognitionResult.innerHTML = '<p class="error">Failed to process attendance.</p>';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            recognitionResult.innerHTML = '<p class="error">Error processing attendance.</p>';
        });
    }
    
    function loadAttendanceData(date) {
        fetch(`/api/attendance_data?date=${date}`)
            .then(response => response.json())
            .then(data => {
                // Clear existing rows
                attendanceList.innerHTML = '';
                
                if (data.length === 0) {
                    const row = document.createElement('tr');
                    row.innerHTML = '<td colspan="3" style="text-align: center;">No attendance records for this date.</td>';
                    attendanceList.appendChild(row);
                } else {
                    // Add each attendance record
                    data.forEach(record => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${record.student_id}</td>
                            <td>${record.name}</td>
                            <td>${formatTime(record.timestamp)}</td>
                        `;
                        attendanceList.appendChild(row);
                    });
                }
            })
            .catch(error => {
                console.error('Error loading attendance data:', error);
                attendanceList.innerHTML = '<tr><td colspan="3" class="error">Error loading attendance data.</td></tr>';
            });
    }
    
    function formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });
    }
    
    function formatTime(isoString) {
        const date = new Date(isoString);
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }
});

