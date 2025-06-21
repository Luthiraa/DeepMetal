#include "output/model.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("starting debug test...\n");
    
    // much smaller test input
    float input[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    
    printf("input created\n");
    
    // try to call predict with minimal data
    printf("calling predict...\n");
    int prediction = predict(input, 1, 1, 10);  // treat as 1x1x10 input
    
    printf("prediction: %d\n", prediction);
    return 0;
}
