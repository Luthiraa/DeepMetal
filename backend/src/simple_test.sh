#!/bin/bash
# simple_test.sh - minimal pytorch to c test

echo "🚀 Simple PyTorch to C Test"
echo "=========================="

# create model if needed
if [ ! -f "models/mnist_linear_model.pth" ]; then
    echo "📦 creating model..."
    python export_model.py --model-type linear --epochs 1 --no-train
fi

# convert to c
echo "📝 converting to c..."
python converter.py models/mnist_linear_model.pth

# quick test
echo "🧪 testing c code..."
cat > quick_test.c << 'EOF'
#include "output/model.h"
#include <stdio.h>
int main() {
    float input[784] = {0.1f};
    int result = predict(input, 28, 28, 1);
    printf("prediction: %d\n", result);
    return 0;
}
EOF

gcc -I output/ quick_test.c output/model.c -o quick_test -lm
./quick_test

echo "✅ done! check output/ for generated files"
rm quick_test quick_test.c