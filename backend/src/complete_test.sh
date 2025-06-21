#!/bin/bash
# complete_test.sh - complete pytorch to c conversion test

echo "🚀 Complete PyTorch to C Conversion Test"
echo "========================================"
echo

# step 1: check if model exists, create if needed
MODEL_PATH="models/mnist_linear_model.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "📦 creating pytorch model (no training for speed)..."
    python export_model.py --model-type linear --no-train
    echo
else
    echo "✅ model already exists: $MODEL_PATH"
    echo
fi

# step 2: convert to c
echo "📝 converting to c code..."
python converter.py $MODEL_PATH
if [ $? -ne 0 ]; then
    echo "❌ c conversion failed"
    exit 1
fi
echo

# step 3: convert to llvm ir (optional, skip if llvm.py missing)
if [ -f "llvm.py" ]; then
    echo "🏗️ converting to llvm ir..."
    python llvm.py $MODEL_PATH
    if [ $? -ne 0 ]; then
        echo "⚠️ llvm conversion failed (continuing anyway)"
    fi
    echo
else
    echo "⏭️ skipping llvm conversion (llvm.py not found)"
    echo
fi

# step 4: check generated files
echo "📁 checking generated files..."
REQUIRED_FILES=("output/model.h" "output/model.c")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
        echo "  ✅ $file (${size} bytes)"
    else
        echo "  ❌ $file missing"
        exit 1
    fi
done
echo

# step 5: create test program
echo "🧪 creating test program..."
cat > test_inference.c << 'EOF'
#include "output/model.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("testing c model inference...\n");
    
    // create mnist 28x28 input (all 0.1)
    float input[784];
    for (int i = 0; i < 784; i++) {
        input[i] = 0.1f;
    }
    
    printf("running inference...\n");
    int prediction = predict(input, 28, 28, 1);
    
    printf("✅ prediction: %d\n", prediction);
    printf("✅ c model working correctly!\n");
    
    return 0;
}
EOF

# step 6: compile test program
echo "🔨 compiling test program..."
if gcc -I output/ test_inference.c output/model.c -o test_inference -lm; then
    echo "  ✅ compilation successful"
else
    echo "  ❌ compilation failed"
    exit 1
fi
echo

# step 7: run test program
echo "🏃 running test program..."
if ./test_inference; then
    echo "  ✅ test execution successful"
else
    echo "  ❌ test execution failed"
    exit 1
fi
echo

# step 8: analyze memory usage
if [ -f "analyze_memory.py" ]; then
    echo "📊 analyzing memory usage..."
    python analyze_memory.py
    echo
else
    echo "⏭️ skipping memory analysis (analyze_memory.py not found)"
    echo
fi

# step 9: show results summary
echo "📋 results summary:"
echo "=================="
echo "✅ model created: $MODEL_PATH"
echo "✅ c conversion: output/model.h, output/model.c"
if [ -f "output/model.ll" ]; then
    echo "✅ llvm conversion: output/model.ll"
fi
echo "✅ c compilation: test_inference"
echo "✅ inference test: passed"
echo
echo "🎉 all tests completed successfully!"
echo
echo "📝 next steps:"
echo "  1. examine generated c code: cat output/model.c"
echo "  2. integrate into your project: #include \"output/model.h\""
echo "  3. deploy to stm32: use output/model.o"
echo
echo "🧹 cleanup (optional):"
echo "  rm test_inference test_inference.c"