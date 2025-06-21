#!/bin/bash
# test_conversion.sh - tests C and LLVM converters

echo "Testing C and LLVM converters"
echo "model: models\mnist_linear_model.pth"
echo

echo "Testing C converter..."
python converter.py models\mnist_linear_model.pth
echo

echo "Testing LLVM converter..."
python llvm.py models\mnist_linear_model.pth
echo

echo "Conversions complete!"
echo "check output/ directory for generated files"
