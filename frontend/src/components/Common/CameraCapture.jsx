import React, { useRef, useEffect, useState, useCallback } from 'react';
import './CameraCapture.css';

const CameraCapture = ({ onCapture, showImage = null, continuousCapture = false }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [cameraActive, setCameraActive] = useState(true);
  const [error, setError] = useState('');
  const [isContinuousCapturing, setIsContinuousCapturing] = useState(false);
  const [captureCount, setCaptureCount] = useState(0);
  
  // Use a ref to track the interval ID to ensure it persists between renders
  const intervalRef = useRef(null);
  
  const MAX_CAPTURES = 60;
  const CAPTURE_DELAY = 300; // 300ms between captures

  // Use useCallback to memoize the stopContinuousCapture function
  const stopContinuousCapture = useCallback(() => {
    console.log('Stopping continuous capture...');
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
      console.log('Interval cleared');
    }
    setIsContinuousCapturing(false);
  }, []);

  useEffect(() => {
    // Start camera when component mounts
    startCamera();

    // Cleanup function to stop camera when component unmounts
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      // Ensure interval is cleared when component unmounts
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      const videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = videoStream;
      }
      setStream(videoStream);
      setCameraActive(true);
      setError('');
    } catch (err) {
      console.error('Error accessing camera:', err);
      setError('Could not access the camera. Please check permissions.');
      setCameraActive(false);
    }
  };

  const captureImage = () => {
    if (!cameraActive) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext('2d');
    
    // Draw the current video frame to the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg');
    
    // Call the passed onCapture function with the image data
    onCapture(imageData);
  };

  const startContinuousCapture = () => {
    // Reset state and counter
    setCaptureCount(0);
    setIsContinuousCapturing(true);

    // Make sure any existing interval is cleared first
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    // Create and store the new interval
    const interval = setInterval(() => {
      if (!cameraActive) {
        stopContinuousCapture();
        return;
      }

      captureImage();
      
      setCaptureCount(prevCount => {
        const newCount = prevCount + 1;
        
        // Stop when we reach the maximum
        if (newCount >= MAX_CAPTURES) {
          stopContinuousCapture();
        }
        
        return newCount;
      });
    }, CAPTURE_DELAY);
    
    // Store in ref so it persists between renders
    intervalRef.current = interval;
    console.log('Started continuous capture with interval ID:', interval);
  };

  return (
    <div className="camera-component">
      {error && <div className="camera-error">{error}</div>}
      
      <div className="camera-container">
        {cameraActive && !showImage ? (
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            width="640" 
            height="480"
          />
        ) : showImage ? (
          <img src={showImage} alt="Captured" width="640" height="480" />
        ) : (
          <div className="camera-placeholder">Camera not available</div>
        )}
        
        <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />
      </div>
      
      {cameraActive && !showImage && (
        <div className="camera-controls">
          {continuousCapture ? (
            <>
              {!isContinuousCapturing ? (
                <button 
                  className="btn camera-btn" 
                  onClick={startContinuousCapture}
                >
                  Start Continuous Capture
                </button>
              ) : (
                <>
                  <div className="capture-progress">
                    <div 
                      className="progress-bar" 
                      style={{ width: `${(captureCount / MAX_CAPTURES) * 100}%` }}
                    ></div>
                    <div className="progress-text">
                      Capturing: {captureCount} / {MAX_CAPTURES} images
                    </div>
                  </div>
                  <button 
                    className="btn camera-btn stop-btn" 
                    onClick={stopContinuousCapture}
                  >
                    Stop Capture
                  </button>
                </>
              )}
            </>
          ) : (
            <button className="btn camera-btn" onClick={captureImage}>
              Capture Image
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default CameraCapture;
