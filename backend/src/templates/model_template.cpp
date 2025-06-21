#include <iostream>
#include <vector>
#include <cmath>

class MNISTModel {
private:
    // Layer 1 weights and bias
    std::vector<std::vector<float>> layer1_weights = {{LAYER1_WEIGHTS}};
    std::vector<float> layer1_bias = {{LAYER1_BIAS}};
    
    // Layer 2 weights and bias
    std::vector<std::vector<float>> layer2_weights = {{LAYER2_WEIGHTS}};
    std::vector<float> layer2_bias = {{LAYER2_BIAS}};
    
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
        // Layer 1: Linear transformation + ReLU
        std::vector<float> layer1_output(layer1_bias.size());
        for (size_t i = 0; i < layer1_bias.size(); ++i) {
            layer1_output[i] = layer1_bias[i];
            for (size_t j = 0; j < input.size(); ++j) {
                layer1_output[i] += input[j] * layer1_weights[i][j];
            }
            layer1_output[i] = relu(layer1_output[i]);
        }
        
        // Layer 2: Linear transformation + Softmax
        std::vector<float> layer2_output(layer2_bias.size());
        for (size_t i = 0; i < layer2_bias.size(); ++i) {
            layer2_output[i] = layer2_bias[i];
            for (size_t j = 0; j < layer1_output.size(); ++j) {
                layer2_output[i] += layer1_output[j] * layer2_weights[i][j];
            }
        }
        
        return softmax(layer2_output);
    }
    
    int predict_class(const std::vector<float>& input) {
        std::vector<float> probabilities = predict(input);
        return std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
    }
};

// Example usage
int main() {
    MNISTModel model;
    
    // Example input (28x28 = 784 features, normalized to [0,1])
    std::vector<float> example_input(784, 0.0f);
    // Fill with actual pixel values...
    
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
