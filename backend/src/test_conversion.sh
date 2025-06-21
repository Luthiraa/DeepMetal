#!/bin/bash
# test_conversion.sh - tests all three conversion approaches

echo "🔄 testing dynamic neural network converters"
echo "model: models/mnist_hybrid_model.pth"
echo

echo "📝 testing c converter..."
python converter.py models/mnist_hybrid_model.pth
echo

echo "🏗️ testing llvm converter..."
python llvm.py models/mnist_hybrid_model.pth
echo

echo "🎯 testing c++ template converter..."
python pytoc.py models/mnist_hybrid_model.pth
echo

echo "✅ all conversions complete!"
echo "check output/ directory for generated files"
