#!/bin/bash
# minimal_test.sh - tests C converter

echo "🔄 testing C converter"
echo

echo "📝 testing C converter..."
if python converter.py models/test_fixed_model.pth; then
    echo "✅ C converter succeeded"
else
    echo "❌ C converter failed"
fi
echo

echo "✅ C conversion test complete!"
echo "for LLVM IR: install llvmlite then run ./test_conversion.sh"
