#!/bin/bash
# startup.sh - Start both Flask backend and React frontend

echo "🚀 Starting DeepMetal Full Stack Application"
echo "=============================================="

# Function to kill background processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $FLASK_PID 2>/dev/null
    kill $REACT_PID 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Set up signal handling
trap cleanup SIGINT SIGTERM

# Check if Python and Node.js are available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not installed"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ Node.js/npm is required but not installed"
    exit 1
fi

# Start Flask backend
echo "🐍 Starting Flask backend on port 5000..."
cd "$(dirname "$0")"
python3 flask_backend.py &
FLASK_PID=$!

# Wait a moment for Flask to start
sleep 3

# Start React frontend
echo "⚛️  Starting React frontend on port 5173..."
cd frontend/react-app
npm run dev &
REACT_PID=$!

echo ""
echo "🎉 Services started successfully!"
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for services to be interrupted
wait
