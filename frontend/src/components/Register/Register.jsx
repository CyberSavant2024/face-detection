import React, { useState } from 'react';
import CameraCapture from '../Common/CameraCapture';
import { registerStudent, trainModel, checkDuplicateFace } from '../../services/api';
import './Register.css';

const Register = () => {
  const [activeTab, setActiveTab] = useState('camera');
  const [studentName, setStudentName] = useState('');
  const [capturedImages, setCapturedImages] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [registrationStatus, setRegistrationStatus] = useState({ message: '', type: '' });
  const [trainingStatus, setTrainingStatus] = useState({ message: '', type: '' });
  const [isRegistering, setIsRegistering] = useState(false);
  const [isTraining, setIsTraining] = useState(false);

  const handleCapture = (imageData) => {
    setCapturedImages(prevImages => [...prevImages, imageData]);
  };

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    setUploadedFiles([...uploadedFiles, ...files]);
  };

  const handleRemoveImage = (index, type) => {
    if (type === 'captured') {
      const newImages = capturedImages.filter((_, i) => i !== index);
      setCapturedImages(newImages);
    } else {
      const newFiles = uploadedFiles.filter((_, i) => i !== index);
      setUploadedFiles(newFiles);
    }
  };

  const handleClearAllImages = () => {
    setCapturedImages([]);
    setUploadedFiles([]);
  };

  // Helper function to read a File object as data URL
  const readFileAsDataURL = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleRegisterStudent = async () => {
    if (!studentName.trim()) {
      setRegistrationStatus({ message: 'Please enter a student name', type: 'error' });
      return;
    }

    if (capturedImages.length === 0 && uploadedFiles.length === 0) {
      setRegistrationStatus({ message: 'Please capture or upload at least one image', type: 'error' });
      return;
    }

    setIsRegistering(true);
    setRegistrationStatus({ message: 'Checking for duplicate registration...', type: '' });

    try {
      // Use the first captured image or the first uploaded file to check
      const imageToCheck = capturedImages.length > 0 ? 
        capturedImages[0] : 
        await readFileAsDataURL(uploadedFiles[0]);

      const duplicateCheck = await checkDuplicateFace(imageToCheck);
      
      if (duplicateCheck.exists) {
        // Check if we have a valid student name
        const studentName = duplicateCheck.student_name || 'Unknown Student';
        
        setRegistrationStatus({ 
          message: `This person is already registered as "${studentName}" (ID: ${duplicateCheck.student_id})`, 
          type: 'error' 
        });
        setIsRegistering(false);
        return;
      } else if (duplicateCheck.message) {
        // If there's a message but no duplicate, show the message
        setRegistrationStatus({ 
          message: duplicateCheck.message, 
          type: 'info' 
        });
        // Allow registration to continue if there was just an outdated face model
      }

      // Continue with registration
      setRegistrationStatus({ message: 'Registering student...', type: '' });

      const response = await registerStudent(studentName, capturedImages, uploadedFiles);
      
      if (response.success) {
        setRegistrationStatus({ 
          message: `Student registered successfully with ID: ${response.student_id}`, 
          type: 'success' 
        });
        
        // Reset form after successful registration
        setStudentName('');
        setCapturedImages([]);
        setUploadedFiles([]);
      } else {
        setRegistrationStatus({ message: 'Failed to register student', type: 'error' });
      }
    } catch (error) {
      setRegistrationStatus({ message: error.message, type: 'error' });
    } finally {
      setIsRegistering(false);
    }
  };

  const handleTrainModel = async () => {
    setIsTraining(true);
    setTrainingStatus({ message: 'Training model... This may take a moment.', type: '' });

    try {
      const response = await trainModel();
      
      if (response.success) {
        setTrainingStatus({ message: 'Model trained successfully!', type: 'success' });
      } else {
        setTrainingStatus({ message: 'Failed to train model', type: 'error' });
      }
    } catch (error) {
      setTrainingStatus({ message: error.message, type: 'error' });
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="register-container">
      <h1>Register Student</h1>
      
      <div className="form-group">
        <label htmlFor="student-name">Student Name:</label>
        <input 
          type="text" 
          id="student-name" 
          value={studentName}
          onChange={(e) => setStudentName(e.target.value)}
          required
        />
      </div>
      
      <div className="tabs">
        <button 
          className={`tab-btn ${activeTab === 'camera' ? 'active' : ''}`}
          onClick={() => setActiveTab('camera')}
        >
          Capture Images
        </button>
        <button 
          className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          Upload Images
        </button>
      </div>
      
      {activeTab === 'camera' && (
        <div className="tab-content">
          <CameraCapture onCapture={handleCapture} continuousCapture={true} />
          <div id="capture-count">
            Total images captured: {capturedImages.length}
            {capturedImages.length >= 60 && (
              <span className="capture-success"> ✓ Great! You have enough images for accurate recognition.</span>
            )}
          </div>
        </div>
      )}
      
      {activeTab === 'upload' && (
        <div className="tab-content">
          <div className="upload-container">
            <div className="form-group">
              <label htmlFor="image-upload">Upload Student Images:</label>
              <input 
                type="file" 
                id="image-upload" 
                accept="image/*" 
                multiple
                onChange={handleFileUpload}
              />
              <p className="help-text">Select multiple images of the student (JPG, PNG)</p>
            </div>
          </div>
        </div>
      )}
      
      {(capturedImages.length > 0 || uploadedFiles.length > 0) && (
        <div className="preview-container">
          <h3>Captured/Uploaded Images</h3>
          
          {capturedImages.length > 0 && (
            <>
              <h4>Captured Images: {capturedImages.length}</h4>
              <div className="image-preview-container">
                {/* Show only the last 8 images to prevent UI overload */}
                {capturedImages.slice(-8).map((image, index) => (
                  <div className="image-preview" key={`captured-${index}`}>
                    <img src={image} alt={`Student ${capturedImages.length - 8 + index + 1}`} />
                    <button 
                      className="delete-btn"
                      onClick={() => handleRemoveImage(capturedImages.length - 8 + index, 'captured')}
                    >
                      &times;
                    </button>
                  </div>
                ))}
              </div>
              {capturedImages.length > 8 && (
                <div className="more-images-note">
                  Showing the last 8 of {capturedImages.length} images
                </div>
              )}
            </>
          )}
          
          {uploadedFiles.length > 0 && (
            <>
              <h4>Uploaded Files</h4>
              <div className="image-preview-container">
                {uploadedFiles.map((file, index) => (
                  <div className="image-preview" key={`uploaded-${index}`}>
                    <div className="file-name">{file.name}</div>
                    <button 
                      className="delete-btn"
                      onClick={() => handleRemoveImage(index, 'uploaded')}
                    >
                      &times;
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}
          
          <button 
            className="btn btn-secondary"
            onClick={handleClearAllImages}
          >
            Clear All Images
          </button>
        </div>
      )}
      
      <div className="form-buttons">
        <button 
          id="register-btn" 
          className="btn" 
          onClick={handleRegisterStudent}
          disabled={isRegistering || (!studentName.trim() || (capturedImages.length === 0 && uploadedFiles.length === 0))}
        >
          {isRegistering ? 'Registering...' : 'Register Student'}
        </button>
        
        <button 
          id="train-btn" 
          className="btn"
          onClick={handleTrainModel}
          disabled={isTraining}
        >
          {isTraining ? 'Training...' : 'Train Model'}
        </button>
      </div>
      
      {registrationStatus.message && (
        <div className={`status-message ${registrationStatus.type}`}>
          {registrationStatus.message}
        </div>
      )}
      
      {trainingStatus.message && (
        <div className={`status-message ${trainingStatus.type}`}>
          {trainingStatus.message}
        </div>
      )}
    </div>
  );
};

export default Register;
