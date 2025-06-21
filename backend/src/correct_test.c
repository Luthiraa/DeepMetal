#include "output/model.h"
#include <stdio.h>

int main() {
    printf("testing with correct input size...\n");
    
    // correct mnist input size: 28x28 = 784 floats
    float input[784];
    
    // initialize with small values
    for (int i = 0; i < 784; i++) {
        input[i] = 0.1f;
    }
    
    printf("calling predict with 28x28 input...\n");
    int prediction = predict(input, 28, 28, 1);  // correct dimensions
    
    printf("prediction: %d\n", prediction);
    return 0;
}
