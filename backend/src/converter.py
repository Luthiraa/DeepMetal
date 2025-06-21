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
    if len(layer_keys) >= 4:
        # Assume first 4 parameters are layer1_weight, layer1_bias, layer2_weight, layer2_bias
        layer1_weight_key = layer_keys[0]
        layer1_bias_key = layer_keys[1] 
        layer2_weight_key = layer_keys[2]
        layer2_bias_key = layer_keys[3]
        
        print(f"Using keys: {layer1_weight_key}, {layer1_bias_key}, {layer2_weight_key}, {layer2_bias_key}")
        
        template = template.replace('{{LAYER1_WEIGHTS}}', format_array(weights[layer1_weight_key].T))
        template = template.replace('{{LAYER1_BIAS}}', format_array(weights[layer1_bias_key]))
        template = template.replace('{{LAYER2_WEIGHTS}}', format_array(weights[layer2_weight_key].T))
        template = template.replace('{{LAYER2_BIAS}}', format_array(weights[layer2_bias_key]))
    else:
        print("Error: Expected at least 4 parameters (2 layers with weights and biases)")
        return

    with open(OUTPUT_PATH, 'w') as f:
        f.write(template)

    print(f"✅ C++ model written to {OUTPUT_PATH}")


if __name__ == '__main__':
    convert_model('../models/MNIST_model.pth')
