#include "output/model.h"
#include <stdio.h>

int main() {
    // mnist 28x28 input (all zeros for test)
    float input[784] = {0.0f};
    
    // run inference
    int prediction = predict(input, 28, 28, 1);
    printf("prediction: %d\n", prediction);
    return 0;
}
