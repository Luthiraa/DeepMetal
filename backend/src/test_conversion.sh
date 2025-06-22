#!/bin/bash
# test_conversion.sh - tests C and LLVM converters

echo "🔄 testing C and LLVM converters"
echo "model: models/mnist_conv_model.pth"
echo

echo "📝 testing C converter..."
python converter.py models/mnist_conv_model.pth
echo

echo "🏗️ testing LLVM converter..."
python llvm.py models/mnist_conv_model.pth
echo

echo "✅ conversions complete!"
echo "check output/ directory for generated files"
