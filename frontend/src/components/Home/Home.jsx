import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { resetSystem } from '../../services/api';
import './Home.css';

const Home = () => {
  const [showResetConfirmation, setShowResetConfirmation] = useState(false);
  const [resetStatus, setResetStatus] = useState({ message: '', type: '' });
  const [isResetting, setIsResetting] = useState(false);

  const handleResetSystem = async () => {
    if (isResetting) return;
    
    setIsResetting(true);
    setResetStatus({ message: 'Resetting system...', type: '' });
    
    try {
      const response = await resetSystem();
      if (response.success) {
        setResetStatus({ message: 'System reset successful!', type: 'success' });
      } else {
        setResetStatus({ message: `Error: ${response.message}`, type: 'error' });
      }
    } catch (error) {
      setResetStatus({ message: error.message, type: 'error' });
    } finally {
      setIsResetting(false);
      setShowResetConfirmation(false);
    }
  };

  return (
    <div className="home-container">
      <h1>Smart Attendance System</h1>
      
      <div className="card-container">
        <div className="card">
          <h2>Take Attendance</h2>
          <p>Capture image or upload photos to record attendance for registered students.</p>
          <Link to="/attendance" className="btn">Go to Attendance</Link>
        </div>
        
        <div className="card">
          <h2>Register Student</h2>
          <p>Register new students by capturing or uploading facial images.</p>
          <Link to="/register" className="btn">Go to Registration</Link>
        </div>
      </div>
      
      <div className="admin-section">
        <h2>Administration</h2>
        <div className="card">
          <h3>Reset System</h3>
          <p className="warning-text">Warning: This will delete all student data, images, and attendance records.</p>
          
          {!showResetConfirmation ? (
            <button 
              className="btn btn-danger" 
              onClick={() => setShowResetConfirmation(true)}
              disabled={isResetting}
            >
              Reset System
            </button>
          ) : (
            <div className="confirmation-box">
              <p>Are you sure? This action cannot be undone.</p>
              <div className="confirmation-buttons">
                <button 
                  className="btn btn-danger" 
                  onClick={handleResetSystem}
                  disabled={isResetting}
                >
                  Yes, Reset Everything
                </button>
                <button 
                  className="btn btn-secondary" 
                  onClick={() => setShowResetConfirmation(false)}
                  disabled={isResetting}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
          
          {resetStatus.message && (
            <div className={`status-message ${resetStatus.type}`}>
              {resetStatus.message}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Home;
