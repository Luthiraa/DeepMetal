#include "model.h"

// Simple 3-layer neural network weights (very small for STM32)
// Layer 0: 784 -> 8 (sparse weights to save memory)
const float linear_w0[LAYER0_OUT_SIZE][LAYER0_IN_SIZE] = {
    {[100] = 0.1f, [200] = 0.2f, [300] = 0.1f, [400] = 0.2f, [500] = 0.1f, [600] = 0.2f, [700] = 0.1f},
    {[150] = 0.2f, [250] = 0.1f, [350] = 0.2f, [450] = 0.1f, [550] = 0.2f, [650] = 0.1f, [750] = 0.2f},
    {[125] = 0.1f, [225] = 0.2f, [325] = 0.1f, [425] = 0.2f, [525] = 0.1f, [625] = 0.2f, [725] = 0.1f},
    {[175] = 0.2f, [275] = 0.1f, [375] = 0.2f, [475] = 0.1f, [575] = 0.2f, [675] = 0.1f, [775] = 0.2f},
    {[110] = 0.1f, [210] = 0.2f, [310] = 0.1f, [410] = 0.2f, [510] = 0.1f, [610] = 0.2f, [710] = 0.1f},
    {[160] = 0.2f, [260] = 0.1f, [360] = 0.2f, [460] = 0.1f, [560] = 0.2f, [660] = 0.1f, [760] = 0.2f},
    {[135] = 0.1f, [235] = 0.2f, [335] = 0.1f, [435] = 0.2f, [535] = 0.1f, [635] = 0.2f, [735] = 0.1f},
    {[185] = 0.2f, [285] = 0.1f, [385] = 0.2f, [485] = 0.1f, [585] = 0.2f, [685] = 0.1f, [783] = 0.2f}
};

const float linear_b0[LAYER0_OUT_SIZE] = {0.1f, -0.1f, 0.2f, -0.2f, 0.1f, 0.0f, -0.1f, 0.2f};

// Layer 2: 8 -> 4
const float linear_w2[LAYER2_OUT_SIZE][LAYER2_IN_SIZE] = {
    {0.3f, -0.2f, 0.1f, 0.4f, -0.1f, 0.2f, 0.3f, -0.3f},
    {-0.1f, 0.4f, -0.3f, 0.2f, 0.1f, -0.2f, 0.4f, 0.1f},
    {0.2f, 0.1f, 0.3f, -0.1f, 0.4f, 0.2f, -0.3f, 0.2f},
    {-0.2f, 0.3f, 0.1f, 0.2f, -0.1f, 0.3f, 0.1f, -0.4f}
};

const float linear_b2[LAYER2_OUT_SIZE] = {0.1f, -0.1f, 0.2f, 0.0f};

// Layer 4: 4 -> 10 (output layer)
const float linear_w4[LAYER4_OUT_SIZE][LAYER4_IN_SIZE] = {
    {0.5f, -0.3f, 0.2f, 0.1f},  // digit 0
    {-0.2f, 0.4f, 0.3f, -0.1f}, // digit 1
    {0.3f, 0.1f, -0.4f, 0.2f},  // digit 2
    {0.1f, -0.2f, 0.3f, 0.4f},  // digit 3
    {-0.3f, 0.2f, 0.1f, -0.2f}, // digit 4
    {0.2f, 0.3f, -0.1f, 0.4f},  // digit 5
    {0.4f, -0.1f, 0.2f, 0.3f},  // digit 6
    {-0.1f, 0.3f, 0.4f, -0.2f}, // digit 7
    {0.3f, 0.2f, -0.3f, 0.1f},  // digit 8
    {-0.2f, -0.1f, 0.4f, 0.3f}  // digit 9
};

const float linear_b4[LAYER4_OUT_SIZE] = {0.1f, -0.1f, 0.2f, 0.0f, -0.2f, 0.1f, 0.3f, -0.1f, 0.2f, 0.0f};

// Simple ReLU activation
float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Predict function
int predict(const float *input, int input_h, int input_w, int input_ch) {
    static float buf[32]; // Small buffer for intermediate results
    
    // Avoid unused parameter warnings
    (void)input_h;
    (void)input_w;
    (void)input_ch;
    
    // Layer 0: Linear (784 -> 8) + ReLU
    for(int i = 0; i < LAYER0_OUT_SIZE; i++) {
        float sum = linear_b0[i];
        for(int j = 0; j < LAYER0_IN_SIZE; j++) {
            sum += linear_w0[i][j] * input[j];
        }
        buf[i] = relu(sum);
    }
    
    // Layer 2: Linear (8 -> 4) + ReLU
    for(int i = 0; i < LAYER2_OUT_SIZE; i++) {
        float sum = linear_b2[i];
        for(int j = 0; j < LAYER2_IN_SIZE; j++) {
            sum += linear_w2[i][j] * buf[j];
        }
        buf[8 + i] = relu(sum); // Store in buf[8-11]
    }
    
    // Layer 4: Linear (4 -> 10) - output layer
    float max_val = -1000.0f;
    int max_idx = 0;
    
    for(int i = 0; i < LAYER4_OUT_SIZE; i++) {
        float sum = linear_b4[i];
        for(int j = 0; j < LAYER4_IN_SIZE; j++) {
            sum += linear_w4[i][j] * buf[8 + j]; // Use buf[8-11]
        }
        
        if(sum > max_val) {
            max_val = sum;
            max_idx = i;
        }
    }
    
    return max_idx;
}
