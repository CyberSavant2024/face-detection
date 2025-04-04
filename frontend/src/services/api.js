import axios from 'axios';

// API base URL - use the proxy in development
const api = axios.create({
  baseURL: '/',
  headers: {
    'Content-Type': 'application/json'
  }
});

// Register a student with captured or uploaded images
export const registerStudent = async (name, capturedImages, uploadedFiles) => {
  const formData = new FormData();
  formData.append('name', name);
  formData.append('image_count', capturedImages.length);

  // Add captured images
  capturedImages.forEach((image, index) => {
    formData.append(`image_${index}`, image);
  });

  // Add uploaded files
  if (uploadedFiles && uploadedFiles.length > 0) {
    for (const file of uploadedFiles) {
      formData.append('uploaded_images', file);
    }
  }

  try {
    const response = await api.post('/api/register_student', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Error registering student');
  }
};

// Train the face recognition model
export const trainModel = async () => {
  try {
    const response = await api.post('/api/train_model');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Error training model');
  }
};

// Take attendance with captured or uploaded image
export const takeAttendance = async (date, imageData, uploadedFile) => {
  const formData = new FormData();
  formData.append('date', date);

  if (imageData) {
    formData.append('image', imageData);
  }

  if (uploadedFile) {
    formData.append('uploaded_image', uploadedFile);
  }

  try {
    const response = await api.post('/api/take_attendance', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Error taking attendance');
  }
};

// Get attendance data for a specific date
export const getAttendanceData = async (date) => {
  try {
    const response = await api.get(`/api/attendance_data?date=${date}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Error fetching attendance data');
  }
};

// Reset the entire system
export const resetSystem = async () => {
  try {
    const response = await api.post('/api/reset_system');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Error resetting system');
  }
};

// Check if a face image already exists in the database
export const checkDuplicateFace = async (imageData) => {
  const formData = new FormData();
  
  if (imageData) {
    formData.append('image', imageData);
  }

  try {
    const response = await api.post('/api/check_duplicate_face', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Error checking duplicate face');
  }
};
