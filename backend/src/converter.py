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
    model = torch.load(pytorch_model_path, map_location='cpu')
    model.eval()

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy()

    with open(TEMPLATE_PATH, 'r') as f:
        template = f.read()

    # Replace placeholders
    template = template.replace('{{LAYER1_WEIGHTS}}', format_array(weights['0.weight'].T))
    template = template.replace('{{LAYER1_BIAS}}', format_array(weights['0.bias']))
    template = template.replace('{{LAYER2_WEIGHTS}}', format_array(weights['2.weight'].T))
    template = template.replace('{{LAYER2_BIAS}}', format_array(weights['2.bias']))

    with open(OUTPUT_PATH, 'w') as f:
        f.write(template)

    print(f"✅ C++ model written to {OUTPUT_PATH}")


if __name__ == '__main__':
    convert_model('backend\models\MNIST_model.pth')
