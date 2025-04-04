/**
 * Face detection utility for Smart Attendance System
 */

class FaceDetector {
    constructor(videoElement, statusElement) {
        this.video = videoElement;
        this.statusElement = statusElement;
        this.faceDetectionActive = false;
        this.lastDetectionTime = 0;
        this.detectionInterval = 300; // Check for faces every 300ms
        this.loadPromise = this.loadDependencies();
    }

    async loadDependencies() {
        // Load face-api.js script
        await this.loadScript('https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js');
        
        // Load models
        await faceapi.nets.tinyFaceDetector.loadFromUri('/static/models');
        console.log('Face detection models loaded');
    }
    
    loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.async = true;
            script.onload = () => resolve();
            script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
            document.head.appendChild(script);
        });
    }
    
    async startDetection() {
        try {
            // Wait for dependencies to be loaded
            await this.loadPromise;
            
            this.faceDetectionActive = true;
            this.detectFace();
        } catch (error) {
            console.error('Error starting face detection:', error);
            this.updateStatus('Face detection unavailable', 'not-detected');
        }
    }
    
    stopDetection() {
        this.faceDetectionActive = false;
        if (this.statusElement) {
            this.statusElement.style.display = 'none';
        }
    }
    
    async detectFace() {
        if (!this.faceDetectionActive || !this.video) return;
        
        const now = Date.now();
        if (now - this.lastDetectionTime > this.detectionInterval) {
            this.lastDetectionTime = now;
            
            try {
                const detection = await faceapi.detectSingleFace(
                    this.video, 
                    new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.5 })
                );
                
                if (detection) {
                    this.updateStatus('Face detected', 'detected');
                } else {
                    this.updateStatus('No face detected', 'not-detected');
                }
            } catch (error) {
                console.error('Face detection error:', error);
            }
        }
        
        // Continue detection loop
        requestAnimationFrame(() => this.detectFace());
    }
    
    updateStatus(message, className) {
        if (this.statusElement) {
            this.statusElement.textContent = message;
            this.statusElement.className = className || '';
            this.statusElement.style.display = 'block';
        }
    }
    
    async captureFace(canvasElement, padding = 50) {
        try {
            await this.loadPromise;
            
            // Detect face in the video stream
            const detection = await faceapi.detectSingleFace(
                this.video, 
                new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.5 })
            );
            
            const context = canvasElement.getContext('2d');
            
            // Draw the full video frame to the canvas
            context.drawImage(this.video, 0, 0, canvasElement.width, canvasElement.height);
            
            if (detection) {
                // Get face box with padding
                const box = detection.box;
                const x = Math.max(0, box.x - padding);
                const y = Math.max(0, box.y - padding);
                const width = Math.min(canvasElement.width - x, box.width + padding * 2);
                const height = Math.min(canvasElement.height - y, box.height + padding * 2);
                
                // Create a temporary canvas for the face crop
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = width;
                tempCanvas.height = height;
                const tempContext = tempCanvas.getContext('2d');
                
                // Draw the face region onto the temp canvas
                tempContext.drawImage(
                    canvasElement, 
                    x, y, width, height,  // Source rectangle
                    0, 0, width, height   // Destination rectangle
                );
                
                // Return the cropped face image as base64
                return {
                    faceDetected: true,
                    imageData: tempCanvas.toDataURL('image/jpeg')
                };
            }
            
            // No face detected, return the full frame
            return {
                faceDetected: false,
                imageData: canvasElement.toDataURL('image/jpeg')
            };
            
        } catch (error) {
            console.error('Error capturing face:', error);
            return {
                faceDetected: false,
                imageData: canvasElement.toDataURL('image/jpeg')
            };
        }
    }
}
