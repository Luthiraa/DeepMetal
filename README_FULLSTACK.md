# DeepMetal MNIST Full Stack Application

A modern web application for MNIST digit recognition with STM32 neural network inference. Upload handwritten digit images and get real-time predictions with generated STM32 C code.

## 🌟 Features

- **Modern React Frontend**: Beautiful, responsive UI built with TypeScript and Tailwind CSS
- **Real-time Image Processing**: Drag-and-drop MNIST image upload with instant preprocessing
- **Neural Network Inference**: Trained PyTorch model running on Flask backend
- **STM32 Code Generation**: Automatic generation of optimized C code for STM32F446RE
- **Hardware Integration**: LED blinking patterns to indicate predicted digits
- **Copy-to-Clipboard**: Easy code copying for embedded development

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend│────│  Flask Backend  │────│   STM32 MCU     │
│   (TypeScript)  │HTTP│    (Python)     │C   │  (ARM Cortex)   │
│                 │    │                 │Code│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Frontend (React + TypeScript)
- Image upload with drag-and-drop
- Real-time prediction display
- Code visualization and copying
- Modern, responsive design

### Backend (Flask + PyTorch)
- Image preprocessing (28x28 grayscale normalization)
- Neural network inference
- STM32 C code generation
- RESTful API endpoints

### Hardware (STM32F446RE)
- Optimized neural network in C
- LED-based output indication
- Real-time embedded inference

## 🚀 Quick Start

### Prerequisites

- **Node.js** (v18+) and npm
- **Python 3.8+** with pip
- **STM32 Development Tools** (optional, for hardware deployment)

### 1. Install Dependencies

```bash
# Install Python backend dependencies
pip install -r flask_requirements.txt

# Install React frontend dependencies
cd frontend/react-app
npm install
cd ../..
```

### 2. Start the Application

```bash
# Start both backend and frontend
./startup.sh
```

Or start services individually:

```bash
# Terminal 1: Start Flask backend
python3 flask_backend.py

# Terminal 2: Start React frontend
cd frontend/react-app
npm run dev
```

### 3. Open in Browser

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5000

## 📱 How to Use

1. **Upload Image**: Drag and drop an MNIST digit image or click to select
2. **View Prediction**: See the predicted digit, confidence, and processing time
3. **Copy Code**: Use the generated C code for STM32 deployment
4. **Deploy to Hardware**: Flash the code to STM32F446RE for real inference

## 🔧 API Endpoints

### `POST /api/process-mnist`
Upload and process MNIST image
- **Input**: Form data with image file
- **Output**: JSON with prediction results and generated C code

### `GET /api/health`
Health check endpoint
- **Output**: Service status and model information

### `GET /api/model-info`
Get neural network model details
- **Output**: Model parameters and architecture info

## 🎯 STM32 Deployment

### Generated Code Features
- **Optimized Neural Network**: Converted PyTorch model to efficient C code
- **LED Output**: Prediction indicated by LED blink count (digit + 1)
- **Memory Efficient**: Optimized for STM32F446RE constraints
- **Real-time**: Millisecond inference times

### Compilation Commands
```bash
# Compile for STM32F446RE
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -O2 -Wall -c main.c -o main.o
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -O2 -Wall -c model.c -o model.o
arm-none-eabi-ld -T linker.ld main.o model.o -o neural_net.elf
arm-none-eabi-objcopy -O binary neural_net.elf neural_net.bin

# Flash to board
st-flash write neural_net.bin 0x8000000
```

## 🎨 Technology Stack

### Frontend
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Lucide Icons** for UI elements
- **Axios** for HTTP requests
- **React Dropzone** for file upload

### Backend
- **Flask** web framework
- **PyTorch** for neural networks
- **PIL** for image processing
- **NumPy** for array operations
- **CORS** for cross-origin requests

### Hardware
- **STM32F446RE** microcontroller
- **ARM Cortex-M4** processor
- **Generated C code** for inference

## 📊 Model Information

- **Dataset**: MNIST handwritten digits (0-9)
- **Architecture**: Hybrid CNN + Linear layers
- **Input**: 28x28 grayscale images
- **Output**: 10-class digit classification
- **Optimization**: Quantized for embedded deployment

## 🛠️ Development

### Project Structure
```
DeepMetal/
├── frontend/react-app/          # React TypeScript frontend
│   ├── src/components/          # React components
│   ├── src/types.ts            # TypeScript interfaces
│   └── package.json            # Frontend dependencies
├── backend/src/                # Python ML backend
│   ├── models.py              # Neural network models
│   ├── converter.py           # C code generation
│   └── export_model.py        # Model training/export
├── flask_backend.py           # Flask API server
├── startup.sh                 # Development startup script
└── README_FULLSTACK.md       # This file
```

### Adding New Features

1. **Frontend**: Add components in `frontend/react-app/src/components/`
2. **Backend**: Extend Flask routes in `flask_backend.py`
3. **Models**: Modify neural networks in `backend/src/models.py`
4. **Hardware**: Update STM32 code generation in `backend/src/converter.py`

## 🐛 Troubleshooting

### Common Issues

**Backend not starting**
- Check Python dependencies: `pip install -r flask_requirements.txt`
- Verify PyTorch installation: `python3 -c "import torch; print(torch.__version__)"`

**Frontend build errors**
- Clear npm cache: `npm cache clean --force`
- Reinstall dependencies: `rm -rf node_modules && npm install`

**CORS errors**
- Ensure backend is running on port 5000
- Check browser console for network errors

**Model loading errors**
- Run model export first: `python3 backend/src/export_model.py --dataset mnist`
- Check model file exists in `backend/src/models/`

## 📄 License

This project is part of the DeepMetal embedded AI framework for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Built with ❤️ for embedded AI and edge computing**
