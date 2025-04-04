import React, { useState, useEffect } from 'react';
import CameraCapture from '../Common/CameraCapture';
import { takeAttendance, getAttendanceData } from '../../services/api';
import './Attendance.css';

const Attendance = () => {
  const [activeTab, setActiveTab] = useState('camera');
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]); // Today's date in YYYY-MM-DD
  const [capturedImage, setCapturedImage] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [attendanceList, setAttendanceList] = useState([]);
  const [recognitionResult, setRecognitionResult] = useState({ message: 'Capture or upload an image to mark attendance', type: '' });
  const [isProcessing, setIsProcessing] = useState(false);

  // Load attendance data on initial load and when date changes
  useEffect(() => {
    loadAttendanceData();
  }, [date]);

  const loadAttendanceData = async () => {
    try {
      const data = await getAttendanceData(date);
      setAttendanceList(data);
    } catch (error) {
      console.error('Error loading attendance data:', error);
    }
  };

  const handleCapture = (imageData) => {
    setCapturedImage(imageData);
  };

  const handleFileUpload = (e) => {
    if (e.target.files.length > 0) {
      setUploadedFile(e.target.files[0]);
    }
  };

  const handleRetake = () => {
    setCapturedImage(null);
  };

  const handleClearUpload = () => {
    setUploadedFile(null);
  };

  const handleProcessAttendance = async () => {
    if (!capturedImage && !uploadedFile) {
      setRecognitionResult({
        message: 'Please capture or upload an image first',
        type: 'error'
      });
      return;
    }
    
    setIsProcessing(true);
    setRecognitionResult({ message: 'Processing attendance...', type: '' });
    
    try {
      const response = await takeAttendance(date, capturedImage, uploadedFile);
      
      if (response.success) {
        if (response.recognized && response.recognized.length > 0) {
          setRecognitionResult({
            message: `Attendance marked for ${response.recognized.length} student(s)`,
            type: 'success'
          });
          // Reload attendance data
          await loadAttendanceData();
        } else {
          setRecognitionResult({
            message: 'No registered students recognized in the image',
            type: 'error'
          });
        }
      } else {
        setRecognitionResult({
          message: 'Failed to process attendance',
          type: 'error'
        });
      }
    } catch (error) {
      setRecognitionResult({
        message: error.message,
        type: 'error'
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const formatDate = (dateString) => {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString('en-US', options);
  };

  const formatTime = (isoString) => {
    const options = { hour: '2-digit', minute: '2-digit' };
    return new Date(isoString).toLocaleTimeString('en-US', options);
  };

  return (
    <div className="attendance-container">
      <h1>Take Attendance</h1>
      
      <div className="form-group">
        <label htmlFor="attendance-date">Date:</label>
        <input 
          type="date" 
          id="attendance-date" 
          value={date}
          onChange={(e) => setDate(e.target.value)}
          required
        />
      </div>
      
      <div className="tabs">
        <button 
          className={`tab-btn ${activeTab === 'camera' ? 'active' : ''}`}
          onClick={() => setActiveTab('camera')}
        >
          Capture Image
        </button>
        <button 
          className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          Upload Image
        </button>
      </div>
      
      {activeTab === 'camera' && (
        <div className="tab-content">
          {!capturedImage ? (
            <CameraCapture onCapture={handleCapture} />
          ) : (
            <div className="preview-container">
              <h3>Preview</h3>
              <div className="captured-image-container">
                <img src={capturedImage} alt="Captured" />
              </div>
              <button className="btn" onClick={handleRetake}>Retake</button>
            </div>
          )}
        </div>
      )}
      
      {activeTab === 'upload' && (
        <div className="tab-content">
          <div className="upload-container">
            <div className="form-group">
              <label htmlFor="attendance-image-upload">Upload Image:</label>
              <input 
                type="file" 
                id="attendance-image-upload"
                accept="image/*"
                onChange={handleFileUpload}
              />
              <p className="help-text">Select an image file (JPG, PNG)</p>
            </div>
            
            {uploadedFile && (
              <div className="preview-container">
                <h3>Selected File</h3>
                <p>{uploadedFile.name}</p>
                <button className="btn btn-secondary" onClick={handleClearUpload}>Clear</button>
              </div>
            )}
          </div>
        </div>
      )}
      
      <div className="form-buttons">
        <button 
          className="btn"
          onClick={handleProcessAttendance}
          disabled={isProcessing || (!capturedImage && !uploadedFile)}
        >
          {isProcessing ? 'Processing...' : 'Process Attendance'}
        </button>
      </div>
      
      <div className={`recognition-result ${recognitionResult.type}`}>
        {recognitionResult.message}
      </div>
      
      <div className="attendance-table-container">
        <h2>Attendance for {formatDate(date)}</h2>
        
        {attendanceList.length > 0 ? (
          <table className="attendance-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {attendanceList.map((record) => (
                <tr key={record.id}>
                  <td>{record.student_id}</td>
                  <td>{record.name}</td>
                  <td>{formatTime(record.timestamp)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="no-records">No attendance records for this date</div>
        )}
      </div>
    </div>
  );
};

export default Attendance;
