#!/bin/bash
# minimal_test.sh - tests converters that don't require llvmlite

echo "🔄 testing available converters"
echo

echo "📝 testing c converter..."
if python converter.py models/test_fixed_model.pth; then
    echo "✅ c converter succeeded"
else
    echo "❌ c converter failed"
fi
echo

echo "🎯 testing c++ template converter..."
if python pytoc.py models/test_fixed_model.pth; then
    echo "✅ c++ converter succeeded"
else
    echo "❌ c++ converter failed"
fi
echo

echo "✅ basic conversion tests complete!"
echo "install llvmlite to test llvm converter: pip install llvmlite"
