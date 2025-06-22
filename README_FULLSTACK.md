# DeepMetal MNIST Full-Stack Application

A complete full-stack application that converts PyTorch MNIST models to optimized STM32 C code and runs them on embedded hardware.

## 🚀 Features

- **React Frontend**: Modern UI with drag-and-drop image upload
- **Flask Backend**: PyTorch model inference and C code generation
- **Model Export**: Automatic PyTorch model loading and conversion
- **STM32 Code Generation**: Complete C code ready for STM32F446RE
- **Real-time Processing**: Upload MNIST images and get instant predictions
- **Code Preview**: View generated STM32 code with syntax highlighting
- **Copy to Clipboard**: Easy code copying for development

## 🏗️ Architecture

```
Frontend (React) ←→ Backend (Flask) ←→ PyTorch Models ←→ STM32 Converter
     ↓                    ↓                    ↓                    ↓
Image Upload      Model Inference      Neural Network      C Code Generation
     ↓                    ↓                    ↓                    ↓
Drag & Drop       Real-time Results    MNIST Recognition   STM32F446RE Code
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- Node.js 16+
- ARM GCC (optional, for compilation)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DeepMetal-1
   ```

2. **Install dependencies**:
   ```bash
   # Install Python dependencies
   pip install -r flask_mnist_requirements.txt
   
   # Install Node.js dependencies
   npm install
   ```

3. **Start the application**:
   ```bash
   # Use the startup script (recommended)
   ./startup.sh
   
   # Or start manually:
   # Terminal 1: python3 flask_mnist_backend.py
   # Terminal 2: npm run dev
   ```

4. **Access the application**:
   - Frontend: http://localhost:5173
   - Backend: http://localhost:5000

## 🔧 Backend Features

### Model Management
- **Automatic Model Loading**: Loads existing PyTorch models from `backend/src/models/`
- **Model Creation**: Creates new models if none exist using `export_model.py`
- **Multiple Architectures**: Supports linear, convolutional, and hybrid models
- **Fallback Models**: Provides simple models if converter is unavailable

### Image Processing
- **MNIST Preprocessing**: Converts uploaded images to 28×28 grayscale format
- **Normalization**: Applies proper MNIST normalization (0-1 range)
- **Real-time Inference**: Runs PyTorch model inference on uploaded images
- **Confidence Scoring**: Provides prediction confidence scores

### C Code Generation
- **Dynamic Converter**: Uses `converter.py` for PyTorch-to-C conversion
- **STM32 Integration**: Generates complete STM32F446RE code
- **LED Feedback**: Code includes LED blinking for digit display
- **Fallback Generation**: Simple C code if converter fails

### API Endpoints

- `GET /api/health` - Health check and service status
- `GET /api/model-info` - Model architecture and parameters
- `POST /api/process-mnist` - Process uploaded MNIST image

## 🌐 Frontend Features

### Image Upload
- **Drag & Drop**: Intuitive file upload interface
- **File Validation**: Supports PNG, JPG, JPEG, GIF, BMP
- **Image Preview**: Shows uploaded image before processing
- **Size Limits**: 16MB maximum file size

### Results Display
- **Prediction Results**: Shows predicted digit and confidence
- **Processing Time**: Displays inference time
- **Model Information**: Shows model type and architecture
- **Real-time Updates**: Live status updates during processing

### Code Preview
- **Syntax Highlighting**: C code with proper formatting
- **Copy to Clipboard**: One-click code copying
- **Complete Code**: Full STM32 project ready for compilation
- **Download Option**: Save generated code to file

## 🎯 Usage

### 1. Upload an Image
- Drag and drop a 28×28 MNIST-style image
- Or click to browse and select a file
- Supported formats: PNG, JPG, JPEG, GIF, BMP

### 2. View Results
- See the predicted digit (0-9)
- Check confidence score
- Review processing time
- Examine model information

### 3. Generate STM32 Code
- View the complete C code
- Copy code to clipboard
- Code includes:
  - Neural network implementation
  - STM32F446RE setup
  - LED control for digit display
  - Main program loop

### 4. Deploy to STM32
- Compile with ARM GCC
- Flash to STM32F446RE Nucleo board
- LED will blink the predicted digit

## 🔍 Testing

Run the test script to verify functionality:

```bash
python3 test_mnist_backend.py
```

Tests include:
- Health check endpoint
- Model information retrieval
- Image processing with test data
- C code generation verification

## 📁 Project Structure

```
DeepMetal-1/
├── flask_mnist_backend.py          # Main Flask backend
├── flask_mnist_requirements.txt    # Python dependencies
├── test_mnist_backend.py          # Backend test script
├── startup.sh                     # Application startup script
├── src/                           # React frontend
│   ├── components/
│   │   ├── ImageUpload.jsx        # Image upload component
│   │   └── AppRouter.jsx          # Simple router
│   └── App.jsx                    # Main app component
├── backend/                       # Backend resources
│   └── src/
│       ├── converter.py           # PyTorch-to-C converter
│       ├── export_model.py        # Model creation utilities
│       └── models/                # Pre-trained models
└── temp_uploads/                  # Temporary file storage
```

## 🛠️ Development

### Adding New Models
1. Place PyTorch models in `backend/src/models/`
2. Models should be compatible with `converter.py`
3. Supported formats: `.pth` files

### Customizing C Code Generation
- Modify `generate_c_code_with_converter()` in `flask_mnist_backend.py`
- Update STM32 setup in generated code
- Add custom LED patterns or UART output

### Frontend Customization
- Edit `src/components/ImageUpload.jsx` for UI changes
- Modify `src/App.jsx` for layout updates
- Update styling in `src/App.css`

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Failed**
   - Check if models exist in `backend/src/models/`
   - Verify PyTorch installation
   - Check model file permissions

2. **Converter Import Error**
   - Ensure `converter.py` is in `backend/src/`
   - Check Python path configuration
   - Fallback code will be generated

3. **Image Processing Errors**
   - Verify image format is supported
   - Check image dimensions (will be resized to 28×28)
   - Ensure proper file permissions

4. **Frontend Connection Issues**
   - Verify backend is running on port 5000
   - Check CORS configuration
   - Ensure no firewall blocking

### Debug Mode

Enable debug logging:

```bash
# Backend debug
export FLASK_DEBUG=1
python3 flask_mnist_backend.py

# Frontend debug
npm run dev -- --debug
```

## 📊 Performance

- **Inference Time**: ~10-50ms per image
- **Model Size**: 1KB - 100KB depending on architecture
- **Memory Usage**: ~50MB for backend, ~100MB for frontend
- **Code Generation**: ~1-5 seconds for complex models

## 🔮 Future Enhancements

- [ ] Support for other datasets (CIFAR-10, Fashion-MNIST)
- [ ] Real-time camera input
- [ ] Model training interface
- [ ] Advanced STM32 configurations
- [ ] WebSocket for real-time updates
- [ ] Model quantization for smaller code size

## 📄 License

This project is part of the DeepMetal (Py2STM) framework for converting PyTorch models to STM32 code.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

**DeepMetal MNIST** - From PyTorch to STM32 in minutes! 🚀
