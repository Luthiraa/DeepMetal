import torch
import numpy as np

TEMPLATE_PATH = 'templates/model_template.cpp'
OUTPUT_PATH = 'output/generated_model.cpp'


def format_array(arr):
    """Formats a NumPy array into a C++ initializer list."""
    if arr.ndim == 1:
        return '{' + ', '.join(f'{x:.6f}' for x in arr) + '}'
    elif arr.ndim == 2:
        return '{' + ',\n '.join(format_array(row) for row in arr) + '}'
    else:
        raise ValueError("Only supports 1D or 2D arrays for now")


def convert_model(pytorch_model_path):
    # Load the state_dict (weights dictionary)
    state_dict = torch.load(pytorch_model_path, map_location='cpu')
    
    # Convert tensors to numpy arrays
    weights = {}
    for name, tensor in state_dict.items():
        weights[name] = tensor.detach().numpy()

    # Print available keys to debug
    print("Available keys in the model:")
    for key in weights.keys():
        print(f"  {key}: {weights[key].shape}")

    with open(TEMPLATE_PATH, 'r') as f:
        template = f.read()

    # Find the correct layer names (adapt based on actual keys)
    layer_keys = list(weights.keys())
      # Try to identify layer weights and biases
    if len(layer_keys) >= 6:  # 3 layers with weights and biases each
        # Map the keys to their respective layers
        layer1_weight_key = 'hidden_0.weight'
        layer1_bias_key = 'hidden_0.bias'
        layer2_weight_key = 'hidden_1.weight'
        layer2_bias_key = 'hidden_1.bias'
        layer3_weight_key = 'output.weight'
        layer3_bias_key = 'output.bias'
        
        print(f"Using keys: {layer1_weight_key}, {layer1_bias_key}, {layer2_weight_key}, {layer2_bias_key}, {layer3_weight_key}, {layer3_bias_key}")
        
        template = template.replace('{{LAYER1_WEIGHTS}}', format_array(weights[layer1_weight_key].T))
        template = template.replace('{{LAYER1_BIAS}}', format_array(weights[layer1_bias_key]))
        template = template.replace('{{LAYER2_WEIGHTS}}', format_array(weights[layer2_weight_key].T))
        template = template.replace('{{LAYER2_BIAS}}', format_array(weights[layer2_bias_key]))
        template = template.replace('{{LAYER3_WEIGHTS}}', format_array(weights[layer3_weight_key].T))
        template = template.replace('{{LAYER3_BIAS}}', format_array(weights[layer3_bias_key]))
    else:
        print("Error: Expected 6 parameters (3 layers with weights and biases each)")
        return

    with open(OUTPUT_PATH, 'w') as f:
        f.write(template)

    print(f"✅ C++ model written to {OUTPUT_PATH}")


if __name__ == '__main__':
    convert_model('../models/MNIST_model.pth')
