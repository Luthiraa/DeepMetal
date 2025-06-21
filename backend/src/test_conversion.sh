#!/bin/bash
# test_conversion.sh - tests C and LLVM converters only

echo "🔄 testing C and LLVM converters"
echo "model: models/mnist_hybrid_model.pth"
echo

echo "📝 testing C converter..."
python converter.py models/mnist_hybrid_model.pth
echo

echo "🏗️ testing LLVM converter..."
python llvm.py models/mnist_hybrid_model.pth
echo

echo "✅ conversions complete!"
echo "check output/ directory for:"
echo "  - model.h, model.c, model.o (C code)"
echo "  - model.ll, model_llvm.o (LLVM IR)"
