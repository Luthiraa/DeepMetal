#!/bin/bash
# startup.sh - Start both Flask backend and React frontend

echo "�� Starting DeepMetal MNIST Full-Stack Application"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed or not in PATH"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -f "flask_mnist_requirements.txt" ]; then
    pip install -r flask_mnist_requirements.txt
else
    echo "⚠️  flask_mnist_requirements.txt not found, using basic requirements"
    pip install flask flask-cors torch torchvision pillow numpy
fi

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Start the Flask backend in the background
echo "🔧 Starting Flask backend..."
python3 flask_mnist_backend.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start the React frontend
echo "🌐 Starting React frontend..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Both services are starting..."
echo "📊 Backend: http://localhost:5000"
echo "🌐 Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait
