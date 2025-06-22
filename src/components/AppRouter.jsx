import React from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import ImageUpload from './ImageUpload';
import Work from '../Work';
import App from '../App';

// Wrapper components to provide navigation functions
const WorkWithNavigation = () => {
  const navigate = useNavigate();
  
  const navigateToUpload = () => navigate('/upload');
  const navigateToApp = () => navigate('/app');
  
  return (
    <Work 
      onNavigateToUpload={navigateToUpload}
      onNavigateToApp={navigateToApp}
    />
  );
};

const AppWithNavigation = () => {
  const navigate = useNavigate();
  
  const navigateToWork = () => navigate('/');
  const navigateToUpload = () => navigate('/upload');
  
  return (
    <App 
      onNavigateToWork={navigateToWork}
      onNavigateToUpload={navigateToUpload}
    />
  );
};

const ImageUploadWithNavigation = () => {
  const navigate = useNavigate();
  
  const navigateToWork = () => navigate('/');
  const navigateToApp = () => navigate('/app');
  
  return (
    <ImageUpload 
      onNavigateToWork={navigateToWork}
      onNavigateToApp={navigateToApp}
    />
  );
};

const AppRouter = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<WorkWithNavigation />} />
        <Route path="/app" element={<AppWithNavigation />} />
        <Route path="/upload" element={<ImageUploadWithNavigation />} />
      </Routes>
    </Router>
  );
};

export default AppRouter; 