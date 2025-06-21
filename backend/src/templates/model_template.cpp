#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

class MNISTModel {
private:
    // Layer 1: Input (784) -> Hidden1 (128)
    std::vector<std::vector<float>> layer1_weights = {{LAYER1_WEIGHTS}};
    std::vector<float> layer1_bias = {{LAYER1_BIAS}};
    
    // Layer 2: Hidden1 (128) -> Hidden2 (64)
    std::vector<std::vector<float>> layer2_weights = {{LAYER2_WEIGHTS}};
    std::vector<float> layer2_bias = {{LAYER2_BIAS}};
    
    // Layer 3: Hidden2 (64) -> Output (10)
    std::vector<std::vector<float>> layer3_weights = {{LAYER3_WEIGHTS}};
    std::vector<float> layer3_bias = {{LAYER3_BIAS}};
    
    // ReLU activation function
    float relu(float x) {
        return std::max(0.0f, x);
    }
    
    // Softmax activation function
    std::vector<float> softmax(const std::vector<float>& input) {
        std::vector<float> output(input.size());
        float max_val = *std::max_element(input.begin(), input.end());
        float sum = 0.0f;
        
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }
        
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] /= sum;
        }
        
        return output;
    }

public:
    std::vector<float> predict(const std::vector<float>& input) {
        // Layer 1: Input -> Hidden1 (784 -> 128)
        std::vector<float> hidden1(128);
        for (int i = 0; i < 128; ++i) {
            hidden1[i] = layer1_bias[i];
            for (int j = 0; j < 784; ++j) {
                hidden1[i] += input[j] * layer1_weights[i][j];
            }
            hidden1[i] = relu(hidden1[i]);
        }
        
        // Layer 2: Hidden1 -> Hidden2 (128 -> 64)
        std::vector<float> hidden2(64);
        for (int i = 0; i < 64; ++i) {
            hidden2[i] = layer2_bias[i];
            for (int j = 0; j < 128; ++j) {
                hidden2[i] += hidden1[j] * layer2_weights[i][j];
            }
            hidden2[i] = relu(hidden2[i]);
        }
        
        // Layer 3: Hidden2 -> Output (64 -> 10)
        std::vector<float> output(10);
        for (int i = 0; i < 10; ++i) {
            output[i] = layer3_bias[i];
            for (int j = 0; j < 64; ++j) {
                output[i] += hidden2[j] * layer3_weights[i][j];
            }
        }
        
        return softmax(output);
    }
    
    int predict_class(const std::vector<float>& input) {
        std::vector<float> probabilities = predict(input);
        return std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
    }
};

int main() {
    MNISTModel model;
    std::vector<float> example_input(784, 0.0f);
    std::vector<float> probabilities = model.predict(example_input);
    int predicted_class = model.predict_class(example_input);

    std::cout << "Predicted class: " << predicted_class << std::endl;
    std::cout << "Probabilities: ";
    for (size_t i = 0; i < probabilities.size(); ++i) {
        std::cout << probabilities[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
