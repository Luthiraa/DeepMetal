import React, { useState, useCallback } from 'react';
import { motion } from 'motion/react';
import { Copy, Upload, FileImage, Code, Download, CheckCircle, AlertCircle } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import FloatingStars from './sparkles';

const ImageUpload = ({ onNavigateToWork, onNavigateToApp }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [processingTime, setProcessingTime] = useState(null);
  const [error, setError] = useState(null);
  const [copySuccess, setCopySuccess] = useState(false);
  const [modelC, setModelC] = useState('');
  const [showCode, setShowCode] = useState(false);
  const [uploadedFilename, setUploadedFilename] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
      setError(null);
    } else {
      setError('Please select a valid image file.');
    }
  };

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  }, []);

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  };

  const processImage = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setError(null);
    setShowCode(false);
    setPrediction(null);
    setUploadProgress(0);
    
    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 100);
    
    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await fetch('http://localhost:5000/api/process-mnist', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data.prediction);
      setConfidence(data.confidence);
      setModelC(data.model_c || '');
      setUploadedFilename(data.uploaded_filename || selectedImage.name);
      setProcessingTime(data.processing_time);
      setShowCode(true);
      setUploadProgress(100);

    } catch (err) {
      setError(err.message || 'An unexpected error occurred.');
      setShowCode(false);
    } finally {
      setIsLoading(false);
      clearInterval(progressInterval);
    }
  };

  const resetForm = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setPrediction(null);
    setConfidence(null);
    setProcessingTime(null);
    setModelC('');
    setUploadedFilename('');
    setShowCode(false);
    setError(null);
    setCopySuccess(false);
    setIsLoading(false);
    setUploadProgress(0);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white overflow--scroll">
      {/* Floating Stars */}
      <FloatingStars count={8} />
      
      {/* Header */}
      <motion.header 
        className="fixed top-0 left-0 w-full z-50 bg-[#0a101f]/50 backdrop-blur-sm py-8 px-8 lg:px-16 flex justify-between items-center"
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <motion.div 
          className="flex-row items-center justify-center flex space-x-4"
          initial={{ x: -50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
        >
          <motion.div 
            className="text-4xl font-bold"
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.2 }}
          >
            Py2STM
          </motion.div>
          <Avatar>
            <AvatarImage src="logo.svg" alt="logo" />
            <AvatarFallback className="bg-blue-600 text-white">P</AvatarFallback>
          </Avatar>
        </motion.div>

        <motion.nav 
          className="hidden md:flex items-center space-x-8 font-bold text-xl"
          initial={{ x: 50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.4, ease: "easeOut" }}
        >
          {onNavigateToWork && (
            <Button
              variant="ghost"
              onClick={onNavigateToWork}
              className="hover:text-blue-400 transition-colors text-xl font-bold"
            >
              Home
            </Button>
          )}
          {onNavigateToApp && (
            <Button
              variant="ghost"
              onClick={onNavigateToApp}
              className="hover:text-blue-400 transition-colors text-xl font-bold"
            >
              Dashboard
            </Button>
          )}
        </motion.nav>
      </motion.header>

      {/* Main Content */}
      <main className="pt-40 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div 
          className="text-center mb-8"
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.6, ease: "easeOut" }}
        >
          <h2 className="text-3xl font-bold text-gray-100 mb-4">
            MNIST Digit Recognition
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Upload an image of a handwritten digit. The prediction will be based on the filename.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <motion.div 
            className="space-y-6"
            initial={{ x: -100, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.8, ease: "easeOut" }}
          >
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-xl text-gray-100 flex items-center gap-2">
                  <Upload className="w-5 h-5" />
                  Upload Image
                </CardTitle>
                <CardDescription className="text-gray-400">
                  Drag and drop your image or click to browse files
                </CardDescription>
              </CardHeader>
              <CardContent>
                {/* Drag & Drop Area */}
                <motion.div
                  className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
                    isDragOver
                      ? 'border-blue-400 bg-blue-400/10'
                      : 'border-gray-600 hover:border-gray-500'
                  }`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {previewUrl ? (
                    <motion.div 
                      className="space-y-4"
                      initial={{ scale: 0.8, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ duration: 0.5 }}
                    >
                      <img
                        src={previewUrl}
                        alt="Preview"
                        className="max-w-full h-48 object-contain mx-auto rounded-lg"
                      />
                      <div className="space-y-2">
                        <p className="text-sm text-gray-400">
                          {selectedImage?.name}
                        </p>
                        <div className="flex justify-center space-x-2">
                          <Button
                            onClick={processImage}
                            disabled={isLoading}
                            variant="gradient"
                            className="font-medium"
                          >
                            {isLoading ? (
                              <>
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                                Processing...
                              </>
                            ) : (
                              <>
                                <Code className="w-4 h-4 mr-2" />
                                Process Image
                              </>
                            )}
                          </Button>
                          <Button
                            onClick={resetForm}
                            variant="outline"
                            className="border-gray-600 text-gray-300 hover:bg-gray-700"
                          >
                            Reset
                          </Button>
                        </div>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div 
                      className="space-y-4"
                      initial={{ y: 20, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      transition={{ duration: 0.5 }}
                    >
                      <motion.div 
                        className="w-16 h-16 bg-gray-700 rounded-full flex items-center justify-center mx-auto"
                        whileHover={{ scale: 1.1, rotate: 5 }}
                        transition={{ duration: 0.2 }}
                      >
                        <FileImage className="w-8 h-8 text-gray-400" />
                      </motion.div>
                      <div>
                        <p className="text-lg font-medium text-gray-200">
                          Drop your image here
                        </p>
                        <p className="text-sm text-gray-400 mt-1">
                          or click to browse files
                        </p>
                      </div>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileInput}
                        className="hidden"
                        id="file-input"
                      />
                      <Dialog>
                        <DialogTrigger asChild>
                          <Button variant="gradient" className="font-medium">
                            <Upload className="w-4 h-4 mr-2" />
                            Choose File
                          </Button>
                        </DialogTrigger>
                        <DialogContent className="bg-gray-800 border-gray-700">
                          <DialogHeader>
                            <DialogTitle className="text-gray-100">Select Image File</DialogTitle>
                            <DialogDescription className="text-gray-400">
                              Choose an image file to upload for digit recognition
                            </DialogDescription>
                          </DialogHeader>
                          <div className="space-y-4">
                            <label htmlFor="file-input" className="block">
                              <div className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center hover:border-gray-500 transition-colors cursor-pointer">
                                <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                                <p className="text-gray-200 font-medium">Click to browse files</p>
                                <p className="text-gray-400 text-sm">or drag and drop</p>
                              </div>
                            </label>
                            <input
                              type="file"
                              accept="image/*"
                              onChange={handleFileInput}
                              className="hidden"
                              id="file-input"
                            />
                          </div>
                        </DialogContent>
                      </Dialog>
                    </motion.div>
                  )}
                </motion.div>

                {/* Progress Bar */}
                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 space-y-2"
                  >
                    <div className="flex justify-between text-sm text-gray-400">
                      <span>Processing...</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <Progress value={uploadProgress} className="h-2" />
                  </motion.div>
                )}

                {/* Error Display */}
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg flex items-center gap-2"
                  >
                    <AlertCircle className="w-4 h-4 text-red-400" />
                    <p className="text-red-400 text-sm">{error}</p>
                  </motion.div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Results Section */}
          <motion.div 
            className="space-y-6"
            initial={{ x: 100, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 1, ease: "easeOut" }}
          >
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-xl text-gray-100 flex items-center gap-2">
                  <CheckCircle className="w-5 h-5" />
                  Prediction Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                {prediction !== null ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="space-y-6"
                  >
                    {/* Prediction Display */}
                    <div className="text-center">
                      <motion.div 
                        className="w-32 h-32 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4"
                        initial={{ scale: 0, rotate: -180 }}
                        animate={{ scale: 1, rotate: 0 }}
                        transition={{ duration: 0.8, type: "spring", bounce: 0.4 }}
                      >
                        <span className="text-6xl font-bold text-white">
                          {prediction}
                        </span>
                      </motion.div>
                      <h4 className="text-2xl font-bold text-gray-100 mb-2">
                        Predicted Digit: {prediction}
                      </h4>
                      <p className="text-gray-400">
                        Confidence: {(confidence * 100).toFixed(1)}%
                      </p>
                    </div>

                    {/* Processing Info */}
                    <Card className="bg-gray-700/50 border-gray-600">
                      <CardContent className="p-4">
                        <h5 className="font-medium text-gray-200 mb-2">Processing Details</h5>
                        <div className="space-y-1 text-sm text-gray-400">
                          <p>Processing Time: {processingTime?.toFixed(2)}ms</p>
                          <p>Model Type: Server-side Prediction</p>
                          <p>Uploaded File: {uploadedFilename}</p>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ) : (
                  <div className="text-center py-12">
                    <motion.div 
                      className="w-16 h-16 bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                    >
                      <FileImage className="w-8 h-8 text-gray-400" />
                    </motion.div>
                    <p className="text-gray-400">
                      Upload an image to see prediction results
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Generated Code Section */}
            {showCode && (
              <motion.div
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
              >
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-xl text-white flex items-center gap-2">
                        <Code className="w-5 h-5" />
                        Generated STM32 Code
                      </CardTitle>
                      <div className="flex items-center gap-2">
                        <Button
                          onClick={() => {
                            navigator.clipboard.writeText(modelC);
                            setCopySuccess(true);
                            setTimeout(() => setCopySuccess(false), 2000);
                          }}
                          variant="outline"
                          size="sm"
                          className="border-gray-600 text-gray-300 hover:bg-gray-700"
                        >
                          <Copy className="w-4 h-4 mr-2" />
                          {copySuccess ? 'Copied!' : 'Copy Code'}
                        </Button>
                        <Button
                          onClick={() => {
                            const blob = new Blob([modelC], { type: 'text/plain' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = 'model_clean.c';
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                          }}
                          variant="outline"
                          size="sm"
                          className="border-gray-600 text-gray-300 hover:bg-gray-700"
                        >
                          <Download className="w-4 h-4 mr-2" />
                          Download
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-gray-900 rounded-lg p-4 max-h-96 overflow-y-auto">
                      <h4 className="font-medium text-white mb-2">model_clean.c</h4>
                      <pre className="text-sm text-gray-300 overflow-x-auto whitespace-pre-wrap">
                        {modelC}
                      </pre>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* Instructions */}
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-lg text-gray-100">How it works</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 text-sm text-gray-400">
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-xs font-bold text-white mt-0.5">
                      1
                    </div>
                    <p>
                      <strong>Upload an Image</strong> of a handwritten digit (0-9).
                    </p>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-xs font-bold text-white mt-0.5">
                      2
                    </div>
                    <p>
                      The backend reads the digit from the filename and generates optimized STM32 C code.
                    </p>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-xs font-bold text-white mt-0.5">
                      3
                    </div>
                    <p>
                      The predicted digit and the generated `main_clean.c` file are displayed.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default ImageUpload; 