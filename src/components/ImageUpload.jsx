import React, { useState, useCallback } from 'react';
import { motion } from 'motion/react';
import { Copy } from 'lucide-react';

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

    } catch (err) {
      setError(err.message || 'An unexpected error occurred.');
      setShowCode(false);
    } finally {
      setIsLoading(false);
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
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white overflow--scroll">
      {/* Header */}
      <header className=" fixed top-0 left-0 w-full z-50 bg-[#0a101f]/50 backdrop-blur-sm py-8 px-8 lg:px-16 flex justify-between items-center">
        <div className="flex-row items-center justify-center flex space-x-4">
          <div className="text-4xl font-bold">Py2STM</div>
          <img src="logo.svg" alt="logo" className="w-10 h-10" />
        </div>

        <nav className="hidden md:flex items-center space-x-8 font-bold text-xl">
          {onNavigateToWork && (
            <button
              onClick={onNavigateToWork}
              className="hover:text-blue-400 transition-colors"
            >
              Home
            </button>
          )}
          {onNavigateToApp && (
            <button
              onClick={onNavigateToApp}
              className="hover:text-blue-400 transition-colors"
            >
              Dashboard
            </button>
          )}
        </nav>
      </header>

      {/* Main Content */}
      <main className="pt-40 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-gray-100 mb-4">
            MNIST Digit Recognition
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Upload an image of a handwritten digit. The prediction will be based on the filename.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-xl font-semibold text-gray-100 mb-4">
                Upload Image
              </h3>
              
              {/* Drag & Drop Area */}
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
                  isDragOver
                    ? 'border-blue-400 bg-blue-400/10'
                    : 'border-gray-600 hover:border-gray-500'
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {previewUrl ? (
                  <div className="space-y-4">
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
                        <button
                          onClick={processImage}
                          disabled={isLoading}
                          className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors duration-200"
                        >
                          {isLoading ? 'Processing...' : 'Process Image'}
                        </button>
                        <button
                          onClick={resetForm}
                          className="bg-gray-600 hover:bg-gray-500 text-white px-4 py-2 rounded-lg transition-colors duration-200"
                        >
                          Reset
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="w-16 h-16 bg-gray-700 rounded-full flex items-center justify-center mx-auto">
                      <svg
                        className="w-8 h-8 text-gray-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        />
                      </svg>
                    </div>
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
                    <label
                      htmlFor="file-input"
                      className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg cursor-pointer transition-colors duration-200 inline-block"
                    >
                      Choose File
                    </label>
                  </div>
                )}
              </div>

              {/* Error Display */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg"
                >
                  <p className="text-red-400 text-sm">{error}</p>
                </motion.div>
              )}
            </div>

            {/* Loading State */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-gray-800 rounded-lg p-6 border border-gray-700"
              >
                <div className="flex items-center space-x-3">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-green-400"></div>
                  <div>
                    <p className="text-gray-200 font-medium">Generating Code...</p>
                    <p className="text-sm text-gray-400">
                      Please wait...
                    </p>
                  </div>
                </div>
              </motion.div>
            )}
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-xl font-semibold text-gray-100 mb-4">
                Prediction Results
              </h3>
              
              {prediction !== null ? (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="space-y-6"
                >
                  {/* Prediction Display */}
                  <div className="text-center">
                    <div className="w-32 h-32 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                      <span className="text-6xl font-bold text-white">
                        {prediction}
                      </span>
                    </div>
                    <h4 className="text-2xl font-bold text-gray-100 mb-2">
                      Predicted Digit: {prediction}
                    </h4>
                    <p className="text-gray-400">
                      Confidence: {(confidence * 100).toFixed(1)}%
                    </p>
                  </div>

                  {/* Processing Info */}
                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-200 mb-2">Processing Details</h5>
                    {/* Processing Details */}
                    {processingTime !== null && (
                      <div className="space-y-1 text-xs text-gray-400">
                        <h4 className="font-semibold text-gray-200 text-sm mb-2">
                          Processing Details
                        </h4>
                        {uploadedFilename && (
                          <div className="flex justify-between">
                            <span>Uploaded File:</span>
                            <span>{uploadedFilename}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </motion.div>
              ) : (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg
                      className="w-8 h-8 text-gray-400"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                  </div>
                  <p className="text-gray-400">
                    Upload an image to see prediction results
                  </p>
                </div>
              )}
            </div>

            {/* Generated Code Section */}
            {showCode && (
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-semibold text-white">Generated STM32 Code</h3>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(modelC);
                      setCopySuccess(true);
                      setTimeout(() => setCopySuccess(false), 2000);
                    }}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                  >
                    <Copy className="w-4 h-4" />
                    {copySuccess ? 'Copied!' : 'Copy model_clean.c'}
                  </button>
                </div>

                {/* Main Clean C Content */}
                <div className="bg-gray-900 rounded-lg p-4">
                  <h4 className="font-medium text-white mb-2">model_clean.c</h4>
                  <pre className="text-sm text-gray-300 overflow-x-auto whitespace-pre-wrap">
                    {modelC}
                  </pre>
                </div>
              </div>
            )}

            {/* Instructions */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-gray-100 mb-3">
                How it works
              </h3>
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
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ImageUpload; 