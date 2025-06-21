# pytoc.py
import torch
import numpy as np
import os
import json
from typing import List, Dict, Any, Tuple

class DynamicTemplateCppConverter:
    def __init__(self, model_path: str, template_dir: str = 'templates', output_dir: str = 'output'):
        self.model_path = model_path
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.layer_configs = []
        self.replacements = {}  # stores template replacement strings
        
        os.makedirs(self.output_dir, exist_ok=True)

    def parse_model_architecture(self):
        """extracts layer configurations from pytorch model"""        # handle pytorch 2.6+ weights_only parameter
        try:
            loaded_data = torch.load(self.model_path, map_location='cpu', weights_only=False)
        except TypeError:
            # fallback for older pytorch versions
            loaded_data = torch.load(self.model_path, map_location='cpu')
        
        # Check if it's a state_dict or a complete model
        if isinstance(loaded_data, dict) and not hasattr(loaded_data, 'eval'):
            # It's a state_dict, parse it directly
            self.parse_state_dict(loaded_data)
            return
        
        # It's a complete model
        model = loaded_data
        model.eval()
        
        layer_idx = 0
        
        # handle both nn.Sequential and direct module iteration
        if isinstance(model, torch.nn.Sequential):
            modules = model
        else:
            modules = [module for module in model.modules() if not isinstance(module, torch.nn.Sequential)]
        
        for module in modules:
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.detach().numpy()
                bias = module.bias.detach().numpy() if module.bias is not None else None
                
                self.layer_configs.append({
                    'type': 'linear',
                    'layer_id': layer_idx,
                    'in_features': weight.shape[1],
                    'out_features': weight.shape[0],
                    'weight': weight.astype(np.float32),
                    'bias': bias.astype(np.float32) if bias is not None else None,
                    'has_bias': bias is not None
                })
                layer_idx += 1
                
            elif isinstance(module, torch.nn.Conv2d):
                weight = module.weight.detach().numpy()
                bias = module.bias.detach().numpy() if module.bias is not None else None
                
                stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
                padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
                
                self.layer_configs.append({
                    'type': 'conv2d',
                    'layer_id': layer_idx,
                    'in_channels': weight.shape[1],
                    'out_channels': weight.shape[0],
                    'kernel_size': (weight.shape[2], weight.shape[3]),
                    'stride': stride,
                    'padding': padding,
                    'weight': weight.astype(np.float32),
                    'bias': bias.astype(np.float32) if bias is not None else None,
                    'has_bias': bias is not None
                })
                layer_idx += 1
                
            elif isinstance(module, torch.nn.ReLU):
                self.layer_configs.append({
                    'type': 'relu',
                    'layer_id': layer_idx
                })
                layer_idx += 1

    def parse_state_dict(self, state_dict):
        """Parse a PyTorch state_dict directly"""
        print("📊 Parsing state_dict...")
        
        # Sort keys to process layers in order
        weight_keys = [k for k in state_dict.keys() if k.endswith('.weight')]
        weight_keys.sort()
        
        print(f"🔍 Found weight keys: {weight_keys}")
        
        layer_idx = 0
        
        for weight_key in weight_keys:
            bias_key = weight_key.replace('.weight', '.bias')
            
            if bias_key in state_dict:
                weight = state_dict[weight_key].detach().numpy().astype(np.float32)
                bias = state_dict[bias_key].detach().numpy().astype(np.float32)
                
                print(f"🔍 {weight_key} shape: {weight.shape}, {bias_key} shape: {bias.shape}")
                
                # Determine layer type based on weight shape
                if len(weight.shape) == 4:  # Conv2D: (out_channels, in_channels, kernel_h, kernel_w)
                    self.layer_configs.append({
                        'type': 'conv2d',
                        'layer_id': layer_idx,
                        'in_channels': weight.shape[1],
                        'out_channels': weight.shape[0],
                        'kernel_size': (weight.shape[2], weight.shape[3]),
                        'stride': (1, 1),  # Default stride
                        'padding': (0, 0),  # Default padding
                        'weight': weight,
                        'bias': bias,
                        'has_bias': True
                    })
                elif len(weight.shape) == 2:  # Linear: (out_features, in_features)
                    self.layer_configs.append({
                        'type': 'linear',
                        'layer_id': layer_idx,
                        'in_features': weight.shape[1],
                        'out_features': weight.shape[0],
                        'weight': weight,
                        'bias': bias,
                        'has_bias': True
                    })
                else:
                    print(f"⚠️  WARNING: Unsupported weight shape {weight.shape} for {weight_key}")
                    continue
                    
                layer_idx += 1

    def format_cpp_array_1d(self, arr: np.ndarray, indent: str = "    ") -> str:
        """formats 1d array as c++ initializer list"""
        elements = [f'{x:.6f}f' for x in arr]
        
        # break into multiple lines if too long
        if len(elements) > 10:
            lines = []
            for i in range(0, len(elements), 10):
                chunk = elements[i:i+10]
                lines.append(indent + ', '.join(chunk))
            return '{\n' + ',\n'.join(lines) + '\n}'
        else:
            return '{' + ', '.join(elements) + '}'

    def format_cpp_array_2d(self, arr: np.ndarray, indent: str = "    ") -> str:
        """formats 2d array as c++ nested initializer list"""
        rows = []
        for i in range(arr.shape[0]):
            row_data = self.format_cpp_array_1d(arr[i], indent + "    ")
            rows.append(indent + row_data)
        return '{\n' + ',\n'.join(rows) + '\n}'

    def format_cpp_array_4d(self, arr: np.ndarray, indent: str = "    ") -> str:
        """formats 4d array as c++ nested initializer list for conv weights"""
        out_channels = []
        for oc in range(arr.shape[0]):
            in_channels = []
            for ic in range(arr.shape[1]):
                kernel_rows = []
                for kh in range(arr.shape[2]):
                    row_data = self.format_cpp_array_1d(arr[oc, ic, kh], indent + "        ")
                    kernel_rows.append(indent + "    " + row_data)
                in_channels.append(indent + "{\n" + ',\n'.join(kernel_rows) + '\n' + indent + "}")
            out_channels.append(indent + "{\n" + ',\n'.join(in_channels) + '\n' + indent + "}")
        return '{\n' + ',\n'.join(out_channels) + '\n}'

    def generate_layer_declarations(self) -> str:
        """generates member variable declarations for all layers"""
        declarations = []
        
        for config in self.layer_configs:
            if config['type'] == 'linear':
                layer_id = config['layer_id']
                in_feat = config['in_features']
                out_feat = config['out_features']
                
                declarations.append(f"    // linear layer {layer_id}: {in_feat} -> {out_feat}")
                declarations.append(f"    std::array<std::array<float, {in_feat}>, {out_feat}> linear_weights_{layer_id};")
                if config['has_bias']:
                    declarations.append(f"    std::array<float, {out_feat}> linear_bias_{layer_id};")
                declarations.append("")
                
            elif config['type'] == 'conv2d':
                layer_id = config['layer_id']
                ic = config['in_channels']
                oc = config['out_channels']
                kh, kw = config['kernel_size']
                
                declarations.append(f"    // conv2d layer {layer_id}: {ic}x{kh}x{kw} -> {oc}")
                declarations.append(f"    std::array<std::array<std::array<std::array<float, {kw}>, {kh}>, {ic}>, {oc}> conv_weights_{layer_id};")
                if config['has_bias']:
                    declarations.append(f"    std::array<float, {oc}> conv_bias_{layer_id};")
                declarations.append("")
        
        return '\n'.join(declarations)

    def generate_layer_initializations(self) -> str:
        """generates constructor initialization list for all layers"""
        initializations = []
        
        for config in self.layer_configs:
            if config['type'] == 'linear':
                layer_id = config['layer_id']
                weight_init = self.format_cpp_array_2d(config['weight'])
                initializations.append(f"        linear_weights_{layer_id}{weight_init}")
                
                if config['has_bias']:
                    bias_init = self.format_cpp_array_1d(config['bias'])
                    initializations.append(f"        linear_bias_{layer_id}{bias_init}")
                    
            elif config['type'] == 'conv2d':
                layer_id = config['layer_id']
                weight_init = self.format_cpp_array_4d(config['weight'])
                initializations.append(f"        conv_weights_{layer_id}{weight_init}")
                
                if config['has_bias']:
                    bias_init = self.format_cpp_array_1d(config['bias'])
                    initializations.append(f"        conv_bias_{layer_id}{bias_init}")
        
        return ',\n'.join(initializations)

    def generate_forward_pass_code(self) -> str:
        """generates forward pass computation code"""
        lines = []
        lines.append("        std::vector<float> current_data = input;")
        lines.append("        int current_h = input_h, current_w = input_w, current_ch = input_ch;")
        lines.append("")
        
        for config in self.layer_configs:
            if config['type'] == 'linear':
                layer_id = config['layer_id']
                in_feat = config['in_features']
                out_feat = config['out_features']
                
                lines.append(f"        // linear layer {layer_id}: {in_feat} -> {out_feat}")
                lines.append(f"        std::vector<float> linear_output_{layer_id}({out_feat});")
                lines.append(f"        for (int i = 0; i < {out_feat}; ++i) {{")
                
                if config['has_bias']:
                    lines.append(f"            float acc = linear_bias_{layer_id}[i];")
                else:
                    lines.append(f"            float acc = 0.0f;")
                    
                lines.append(f"            for (int j = 0; j < {in_feat}; ++j) {{")
                lines.append(f"                acc += current_data[j] * linear_weights_{layer_id}[i][j];")
                lines.append(f"            }}")
                lines.append(f"            linear_output_{layer_id}[i] = acc;")
                lines.append(f"        }}")
                lines.append(f"        current_data = std::move(linear_output_{layer_id});")
                lines.append("")
                
            elif config['type'] == 'conv2d':
                layer_id = config['layer_id']
                ic = config['in_channels']
                oc = config['out_channels']
                kh, kw = config['kernel_size']
                stride_h, stride_w = config['stride']
                pad_h, pad_w = config['padding']
                
                lines.append(f"        // conv2d layer {layer_id}: {ic}x{kh}x{kw} -> {oc}")
                lines.append(f"        int out_h_{layer_id} = (current_h + 2*{pad_h} - {kh}) / {stride_h} + 1;")
                lines.append(f"        int out_w_{layer_id} = (current_w + 2*{pad_w} - {kw}) / {stride_w} + 1;")
                lines.append(f"        std::vector<float> conv_output_{layer_id}({oc} * out_h_{layer_id} * out_w_{layer_id});")
                lines.append("")
                lines.append(f"        for (int oc = 0; oc < {oc}; ++oc) {{")
                lines.append(f"            for (int oh = 0; oh < out_h_{layer_id}; ++oh) {{")
                lines.append(f"                for (int ow = 0; ow < out_w_{layer_id}; ++ow) {{")
                
                if config['has_bias']:
                    lines.append(f"                    float acc = conv_bias_{layer_id}[oc];")
                else:
                    lines.append(f"                    float acc = 0.0f;")
                    
                lines.append(f"                    for (int ic = 0; ic < {ic}; ++ic) {{")
                lines.append(f"                        for (int kh = 0; kh < {kh}; ++kh) {{")
                lines.append(f"                            for (int kw = 0; kw < {kw}; ++kw) {{")
                lines.append(f"                                int ih = oh * {stride_h} - {pad_h} + kh;")
                lines.append(f"                                int iw = ow * {stride_w} - {pad_w} + kw;")
                lines.append(f"                                if (ih >= 0 && ih < current_h && iw >= 0 && iw < current_w) {{")
                lines.append(f"                                    int input_idx = ic * current_h * current_w + ih * current_w + iw;")
                lines.append(f"                                    acc += current_data[input_idx] * conv_weights_{layer_id}[oc][ic][kh][kw];")
                lines.append(f"                                }}")
                lines.append(f"                            }}")
                lines.append(f"                        }}")
                lines.append(f"                    }}")
                lines.append(f"                    int output_idx = oc * out_h_{layer_id} * out_w_{layer_id} + oh * out_w_{layer_id} + ow;")
                lines.append(f"                    conv_output_{layer_id}[output_idx] = acc;")
                lines.append(f"                }}")
                lines.append(f"            }}")
                lines.append(f"        }}")
                lines.append(f"        current_data = std::move(conv_output_{layer_id});")
                lines.append(f"        current_h = out_h_{layer_id}; current_w = out_w_{layer_id}; current_ch = {oc};")
                lines.append("")
                
            elif config['type'] == 'relu':
                layer_id = config['layer_id']
                lines.append(f"        // relu activation {layer_id}")
                lines.append(f"        for (size_t i = 0; i < current_data.size(); ++i) {{")
                lines.append(f"            current_data[i] = std::max(0.0f, current_data[i]);")
                lines.append(f"        }}")
                lines.append("")
        
        lines.append("        return current_data;")
        return '\n'.join(lines)

    def generate_input_size_calculation(self) -> str:
        """calculates input size requirements based on first layer"""
        if not self.layer_configs:
            return "784"  # default mnist size
            
        first_layer = self.layer_configs[0]
        if first_layer['type'] == 'linear':
            return str(first_layer['in_features'])
        elif first_layer['type'] == 'conv2d':
            # for conv layers, calculate total input size for a typical input
            # assume 28x28 input for mnist-like models
            return "28 * 28 * 1"  # typical mnist input
        else:
            return "784"  # fallback

    def create_dynamic_template(self) -> str:
        """creates complete c++ template with dynamic architecture"""
        # determine input dimensions based on first layer
        input_size_calc = self.generate_input_size_calculation()
        
        template = f'''#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

class DynamicNeuralNetwork {{
private:
{self.generate_layer_declarations()}
    // relu activation function
    float relu(float x) {{
        return std::max(0.0f, x);
    }}
    
    // softmax activation function
    std::vector<float> softmax(const std::vector<float>& input) {{
        std::vector<float> output(input.size());
        float max_val = *std::max_element(input.begin(), input.end());
        float sum = 0.0f;
        
        for (size_t i = 0; i < input.size(); ++i) {{
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }}
        
        for (size_t i = 0; i < output.size(); ++i) {{
            output[i] /= sum;
        }}
        
        return output;
    }}

public:
    // constructor with weight initialization
    DynamicNeuralNetwork() :
{self.generate_layer_initializations()}
    {{}}
    
    // forward pass prediction
    std::vector<float> predict(const std::vector<float>& input, int input_h = 28, int input_w = 28, int input_ch = 1) {{
{self.generate_forward_pass_code()}
    }}
    
    // classification prediction (returns class index)
    int predict_class(const std::vector<float>& input, int input_h = 28, int input_w = 28, int input_ch = 1) {{
        std::vector<float> probabilities = predict(input, input_h, input_w, input_ch);
        return std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
    }}
}};

// usage example
int main() {{
    DynamicNeuralNetwork model;
    
    // create example input (use fixed dimensions for typical mnist input)
    const int input_h = 28, input_w = 28, input_ch = 1;
    std::vector<float> example_input({input_size_calc}, 0.0f);
    
    // get predictions
    std::vector<float> probabilities = model.predict(example_input, input_h, input_w, input_ch);
    int predicted_class = model.predict_class(example_input, input_h, input_w, input_ch);
    
    std::cout << "predicted class: " << predicted_class << std::endl;
    std::cout << "probabilities: ";
    for (size_t i = 0; i < probabilities.size(); ++i) {{
        std::cout << probabilities[i] << " ";
    }}
    std::cout << std::endl;
    
    return 0;
}}'''
        return template

    def save_model_config(self):
        """saves model configuration as json for debugging"""
        config_data = {
            'num_layers': len(self.layer_configs),
            'layers': []
        }
        
        for config in self.layer_configs:
            layer_info = {
                'type': config['type'],
                'layer_id': config['layer_id']
            }
            
            if config['type'] == 'linear':
                layer_info.update({
                    'in_features': config['in_features'],
                    'out_features': config['out_features'],
                    'has_bias': config['has_bias']
                })
            elif config['type'] == 'conv2d':
                layer_info.update({
                    'in_channels': config['in_channels'],
                    'out_channels': config['out_channels'],
                    'kernel_size': config['kernel_size'],
                    'stride': config['stride'],
                    'padding': config['padding'],
                    'has_bias': config['has_bias']
                })
            
            config_data['layers'].append(layer_info)
        
        config_file = os.path.join(self.output_dir, 'model_config.json')
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"📋 model configuration saved to: {config_file}")

    def compile_cpp(self):
        """compiles generated c++ code"""
        cpp_file = os.path.join(self.output_dir, 'dynamic_model.cpp')
        exe_file = os.path.join(self.output_dir, 'dynamic_model')
        
        # try different compilers for stm32f446re
        compiler_options = [
            # stm32 cross-compiler (preferred)
            {
                'cmd': 'arm-none-eabi-g++',
                'args': [
                    '-std=c++17',           # modern c++ standard
                    '-mcpu=cortex-m4',      # cortex-m4 cpu
                    '-mthumb',              # thumb instruction set
                    '-mfloat-abi=hard',     # hardware floating point
                    '-mfpu=fpv4-sp-d16',    # stm32f446re fpu
                    '-DSTM32F446xx',        # stm32f446 define
                    '-O3',                  # maximum optimization
                    '-ffunction-sections',  # linker optimization
                    '-fdata-sections',      # linker optimization
                    cpp_file,
                    '-o', exe_file + '_stm32.elf'
                ]
            },
            # native compilers (for testing)
            {
                'cmd': 'clang++',
                'args': [
                    '-std=c++17',           # modern c++ standard
                    '-O3',                  # maximum optimization
                    '-march=native',        # optimize for current cpu
                    '-ffast-math',          # fast math optimizations
                    cpp_file,
                    '-o', exe_file
                ]
            },
            {
                'cmd': 'g++',
                'args': [
                    '-std=c++17',           # modern c++ standard
                    '-O3',                  # maximum optimization
                    '-march=native',        # optimize for current cpu
                    '-ffast-math',          # fast math optimizations
                    cpp_file,
                    '-o', exe_file
                ]
            }
        ]
        
        for compiler_option in compiler_options:
            compiler = compiler_option['cmd']
            args = compiler_option['args']
            
            try:
                import subprocess
                result = subprocess.run([compiler] + args, 
                                      check=True, capture_output=True, text=True)
                
                if 'stm32' in args[-1]:
                    print(f"✅ compiled c++ for stm32f446re with {compiler}: {args[-1]}")
                else:
                    print(f"✅ compiled c++ with {compiler}: {args[-1]}")
                return
                
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                if isinstance(e, subprocess.CalledProcessError):
                    print(f"⚠️ {compiler} compilation failed: {e.stderr}")
                else:
                    print(f"⚠️ {compiler} not found")
                continue
        
        print("💡 no suitable c++ compiler found")
        print("📝 generated c++ code is available in output/dynamic_model.cpp")
        print("🔧 install compilers:")
        print("   stm32: sudo apt install gcc-arm-none-eabi")
        print("   native: sudo apt install clang++ g++")
        print("   or manually compile: g++ -std=c++17 -O3 output/dynamic_model.cpp -o model")

    def convert(self):
        """main template conversion pipeline"""
        print("🔍 parsing model architecture...")
        self.parse_model_architecture()
        
        print(f"📋 found {len(self.layer_configs)} layers:")
        for config in self.layer_configs:
            if config['type'] == 'linear':
                print(f"  - linear {config['layer_id']}: {config['in_features']} -> {config['out_features']}")
            elif config['type'] == 'conv2d':
                print(f"  - conv2d {config['layer_id']}: {config['in_channels']}x{config['kernel_size']} -> {config['out_channels']}")
            elif config['type'] == 'relu':
                print(f"  - relu {config['layer_id']}")
        
        print("📝 generating dynamic c++ template...")
        template_content = self.create_dynamic_template()
        
        cpp_file = os.path.join(self.output_dir, 'dynamic_model.cpp')
        with open(cpp_file, 'w') as f:
            f.write(template_content)
        print(f"💾 c++ model written to: {cpp_file}")
        
        print("📋 saving model configuration...")
        self.save_model_config()
        
        print("🔨 compiling c++ code...")
        self.compile_cpp()
        
        print("✅ template conversion complete!")

if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'model.pth'
    converter = DynamicTemplateCppConverter(model_path)
    converter.convert()