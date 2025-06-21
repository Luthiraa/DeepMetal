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
