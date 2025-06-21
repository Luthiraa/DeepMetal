# converter.py
import torch
import numpy as np
import os
import subprocess
from typing import List, Dict, Any, Tuple

class DynamicPyToCConverter:
    def __init__(self, model_path: str, output_dir: str = 'output'):
        self.model_path = model_path
        self.output_dir = output_dir
        self.layer_configs = []  # stores layer configuration dictionaries
        self.max_buffer_size = 8192  # maximum buffer size for intermediate results
        os.makedirs(self.output_dir, exist_ok=True)

    def parse_model_architecture(self):
        """extracts layer configurations from pytorch model"""
        # handle pytorch 2.6+ weights_only parameter
        try:
            model = torch.load(self.model_path, map_location='cpu', weights_only=False)
        except TypeError:
            # fallback for older pytorch versions
            model = torch.load(self.model_path, map_location='cpu')
        model.eval()
        
        layer_idx = 0
        
        # handle both nn.Sequential and direct module iteration
        if isinstance(model, torch.nn.Sequential):
            modules = model
        else:
            modules = [module for module in model.modules() if not isinstance(module, torch.nn.Sequential)]
        
        for module in modules:
            if isinstance(module, torch.nn.Linear):
                # extract linear layer parameters
                weight = module.weight.detach().numpy()  # shape: [out_features, in_features]
                bias = module.bias.detach().numpy() if module.bias is not None else None
                
                self.layer_configs.append({
                    'type': 'linear',
                    'layer_id': layer_idx,
                    'in_features': weight.shape[1],
                    'out_features': weight.shape[0],
                    'weight': weight.T,  # transpose to [in_features, out_features] for c indexing
                    'bias': bias,
                    'has_bias': bias is not None
                })
                layer_idx += 1
                
            elif isinstance(module, torch.nn.Conv2d):
                # extract conv2d parameters
                weight = module.weight.detach().numpy()  # shape: [out_channels, in_channels, kernel_h, kernel_w]
                bias = module.bias.detach().numpy() if module.bias is not None else None
                
                # handle stride and padding (can be int or tuple)
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
                    'weight': weight,
                    'bias': bias,
                    'has_bias': bias is not None
                })
                layer_idx += 1
                
            elif isinstance(module, torch.nn.ReLU):
                self.layer_configs.append({
                    'type': 'relu',
                    'layer_id': layer_idx
                })
                layer_idx += 1

    def format_array_1d(self, arr: np.ndarray) -> str:
        """formats 1d numpy array as c initializer list"""
        return '{' + ', '.join(f'{float(x):.6f}f' for x in arr) + '}'

    def format_array_2d(self, arr: np.ndarray) -> str:
        """formats 2d numpy array as c initializer list"""
        rows = []
        for row in arr:
            rows.append(self.format_array_1d(row))
        return '{\n    ' + ',\n    '.join(rows) + '\n}'

    def format_array_4d(self, arr: np.ndarray) -> str:
        """formats 4d numpy array for conv weights [out_ch][in_ch][kh][kw]"""
        out_channels = []
        for oc in range(arr.shape[0]):
            in_channels = []
            for ic in range(arr.shape[1]):
                kernel_rows = []
                for kh in range(arr.shape[2]):
                    kernel_row = '{' + ', '.join(f'{float(arr[oc, ic, kh, kw]):.6f}f' for kw in range(arr.shape[3])) + '}'
                    kernel_rows.append(kernel_row)
                in_channels.append('{\n        ' + ',\n        '.join(kernel_rows) + '\n    }')
            out_channels.append('{\n    ' + ',\n    '.join(in_channels) + '\n}')
        return '{\n' + ',\n'.join(out_channels) + '\n}'

    def generate_header_file(self) -> str:
        """generates model.h with layer size macros and extern declarations"""
        lines = ['#ifndef MODEL_H', '#define MODEL_H', '']
        
        # generate size macros
        for config in self.layer_configs:
            if config['type'] == 'linear':
                lines.append(f"#define LAYER{config['layer_id']}_IN_SIZE {config['in_features']}")
                lines.append(f"#define LAYER{config['layer_id']}_OUT_SIZE {config['out_features']}")
            elif config['type'] == 'conv2d':
                lines.append(f"#define LAYER{config['layer_id']}_IN_CH {config['in_channels']}")
                lines.append(f"#define LAYER{config['layer_id']}_OUT_CH {config['out_channels']}")
                lines.append(f"#define LAYER{config['layer_id']}_KH {config['kernel_size'][0]}")
                lines.append(f"#define LAYER{config['layer_id']}_KW {config['kernel_size'][1]}")
        
        lines.append(f'#define MAX_BUFFER_SIZE {self.max_buffer_size}')
        lines.append(f'#define NUM_LAYERS {len(self.layer_configs)}')
        lines.append('')
        
        # generate extern declarations
        for config in self.layer_configs:
            if config['type'] == 'linear':
                lines.append(f"extern const float linear_w{config['layer_id']}[LAYER{config['layer_id']}_OUT_SIZE][LAYER{config['layer_id']}_IN_SIZE];")
                if config['has_bias']:
                    lines.append(f"extern const float linear_b{config['layer_id']}[LAYER{config['layer_id']}_OUT_SIZE];")
            elif config['type'] == 'conv2d':
                lines.append(f"extern const float conv_w{config['layer_id']}[LAYER{config['layer_id']}_OUT_CH][LAYER{config['layer_id']}_IN_CH][LAYER{config['layer_id']}_KH][LAYER{config['layer_id']}_KW];")
                if config['has_bias']:
                    lines.append(f"extern const float conv_b{config['layer_id']}[LAYER{config['layer_id']}_OUT_CH];")
        
        lines.extend(['', 'int predict(const float *input, int input_h, int input_w, int input_ch);', '#endif // MODEL_H'])
        return '\n'.join(lines)

    def generate_linear_layer_code(self, config: Dict[str, Any]) -> Tuple[str, str]:
        """generates weight definitions and computation code for linear layer"""
        layer_id = config['layer_id']
        
        # weight and bias definitions
        weight_def = f"const float linear_w{layer_id}[LAYER{layer_id}_OUT_SIZE][LAYER{layer_id}_IN_SIZE] = {self.format_array_2d(config['weight'].T)};"
        bias_def = f"const float linear_b{layer_id}[LAYER{layer_id}_OUT_SIZE] = {self.format_array_1d(config['bias'])};" if config['has_bias'] else ""
        
        definitions = weight_def + '\n' + bias_def if config['has_bias'] else weight_def
        
        # computation code
        bias_init = f"linear_b{layer_id}[i]" if config['has_bias'] else "0.0f"
        computation = f"""
    // linear layer {layer_id}: {config['in_features']} -> {config['out_features']}
    for (int i = 0; i < LAYER{layer_id}_OUT_SIZE; i++) {{
        float acc = {bias_init};
        for (int j = 0; j < LAYER{layer_id}_IN_SIZE; j++) {{
            acc += prev[j] * linear_w{layer_id}[i][j];
        }}
        nxt[i] = acc;
    }}
    prev_size = LAYER{layer_id}_OUT_SIZE;"""
        
        return definitions, computation

    def generate_conv2d_layer_code(self, config: Dict[str, Any]) -> Tuple[str, str]:
        """generates weight definitions and computation code for conv2d layer"""
        layer_id = config['layer_id']
        
        # weight and bias definitions
        weight_def = f"const float conv_w{layer_id}[LAYER{layer_id}_OUT_CH][LAYER{layer_id}_IN_CH][LAYER{layer_id}_KH][LAYER{layer_id}_KW] = {self.format_array_4d(config['weight'])};"
        bias_def = f"const float conv_b{layer_id}[LAYER{layer_id}_OUT_CH] = {self.format_array_1d(config['bias'])};" if config['has_bias'] else ""
        
        definitions = weight_def + '\n' + bias_def if config['has_bias'] else weight_def
        
        # computation code with stride and padding
        stride_h, stride_w = config['stride']
        pad_h, pad_w = config['padding']
        bias_init = f"conv_b{layer_id}[oc]" if config['has_bias'] else "0.0f"
        
        computation = f"""
    // conv2d layer {layer_id}: {config['in_channels']}x{config['kernel_size'][0]}x{config['kernel_size'][1]} -> {config['out_channels']}
    int out_h_{layer_id} = (prev_h + 2*{pad_h} - LAYER{layer_id}_KH) / {stride_h} + 1;
    int out_w_{layer_id} = (prev_w + 2*{pad_w} - LAYER{layer_id}_KW) / {stride_w} + 1;
    
    for (int oc = 0; oc < LAYER{layer_id}_OUT_CH; oc++) {{
        for (int oh = 0; oh < out_h_{layer_id}; oh++) {{
            for (int ow = 0; ow < out_w_{layer_id}; ow++) {{
                float acc = {bias_init};
                for (int ic = 0; ic < LAYER{layer_id}_IN_CH; ic++) {{
                    for (int kh = 0; kh < LAYER{layer_id}_KH; kh++) {{
                        for (int kw = 0; kw < LAYER{layer_id}_KW; kw++) {{
                            int ih = oh * {stride_h} - {pad_h} + kh;
                            int iw = ow * {stride_w} - {pad_w} + kw;
                            if (ih >= 0 && ih < prev_h && iw >= 0 && iw < prev_w) {{
                                int prev_idx = ic * prev_h * prev_w + ih * prev_w + iw;
                                acc += prev[prev_idx] * conv_w{layer_id}[oc][ic][kh][kw];
                            }}
                        }}
                    }}
                }}
                int out_idx = oc * out_h_{layer_id} * out_w_{layer_id} + oh * out_w_{layer_id} + ow;
                nxt[out_idx] = acc;
            }}
        }}
    }}
    prev_h = out_h_{layer_id}; prev_w = out_w_{layer_id}; prev_ch = LAYER{layer_id}_OUT_CH;
    prev_size = prev_ch * prev_h * prev_w;"""
        
        return definitions, computation

    def generate_relu_layer_code(self) -> str:
        """generates computation code for relu activation"""
        return """
    // relu activation
    for (int i = 0; i < prev_size; i++) {
        nxt[i] = prev[i] > 0.0f ? prev[i] : 0.0f;
    }"""

    def generate_source_file(self) -> str:
        """generates complete model.c source file"""
        lines = ['#include "model.h"', '']
        
        # generate all weight/bias definitions
        for config in self.layer_configs:
            if config['type'] == 'linear':
                definitions, _ = self.generate_linear_layer_code(config)
                lines.append(definitions)
            elif config['type'] == 'conv2d':
                definitions, _ = self.generate_conv2d_layer_code(config)
                lines.append(definitions)
        
        lines.append('')
        
        # generate predict function
        lines.extend([
            'int predict(const float *input, int input_h, int input_w, int input_ch) {',
            '    const float *prev = input;',
            '    static float buf1[MAX_BUFFER_SIZE], buf2[MAX_BUFFER_SIZE];',
            '    float *nxt = buf1;',
            '    int prev_size = input_h * input_w * input_ch;',
            '    int prev_h = input_h, prev_w = input_w, prev_ch = input_ch;',
            ''
        ])
        
        # generate computation for each layer
        for i, config in enumerate(self.layer_configs):
            if config['type'] == 'linear':
                _, computation = self.generate_linear_layer_code(config)
                lines.append(computation)
            elif config['type'] == 'conv2d':
                _, computation = self.generate_conv2d_layer_code(config)
                lines.append(computation)
            elif config['type'] == 'relu':
                computation = self.generate_relu_layer_code()
                lines.append(computation)
            
            # swap buffers after each layer (unique variable names)
            lines.extend([
                '    // swap buffers',
                f'    float *tmp_{i} = (float*)prev; prev = nxt; nxt = tmp_{i};',
                ''
            ])
        
        # final argmax for classification
        lines.extend([
            '    // argmax for classification',
            '    int max_i = 0;',
            '    float max_v = prev[0];',
            '    for (int i = 1; i < prev_size; i++) {',
            '        if (prev[i] > max_v) {',
            '            max_v = prev[i];',
            '            max_i = i;',
            '        }',
            '    }',
            '    return max_i;',
            '}'
        ])
        
        return '\n'.join(lines)

    def compile_to_arm_cortex_m4(self):
        """compiles generated c code to arm cortex-m4 object file for stm32f446re"""
        c_file = os.path.join(self.output_dir, 'model.c')
        obj_file = os.path.join(self.output_dir, 'model.o')
        
        # stm32f446re-specific compiler settings
        cmd = [
            'arm-none-eabi-gcc',            # use arm-none-eabi-gcc for stm32
            '-mcpu=cortex-m4',              # cortex-m4 cpu
            '-mthumb',                      # thumb instruction set
            '-mfloat-abi=hard',             # hardware floating point
            '-mfpu=fpv4-sp-d16',            # floating point unit for stm32f446re
            '-DSTM32F446xx',                # stm32f446 define
            '-O3',                          # maximum optimization
            '-ffast-math',                  # fast math optimizations
            '-ffunction-sections',          # separate functions for linker optimization
            '-fdata-sections',              # separate data for linker optimization
            '-c', c_file,                   # compile only
            '-o', obj_file                  # output object file
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ compiled c to stm32f446re object: {obj_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ stm32 compilation failed: {e.stderr}")
            # fallback to clang if arm-none-eabi-gcc not available
            print("🔄 trying fallback compilation with clang...")
            self.compile_fallback_clang()
        except FileNotFoundError:
            print("❌ arm-none-eabi-gcc not found")
            print("🔄 trying fallback compilation with clang...")
            self.compile_fallback_clang()
    
    def compile_fallback_clang(self):
        """fallback compilation with clang for testing"""
        c_file = os.path.join(self.output_dir, 'model.c')
        obj_file = os.path.join(self.output_dir, 'model_fallback.o')
        
        cmd = [
            'clang',
            '--target=armv7em-none-eabi',   # cortex-m4 target
            '-mcpu=cortex-m4',              # specific cpu
            '-mthumb',                      # thumb instruction set
            '-O3',                          # maximum optimization
            '-c', c_file,                   # compile only
            '-o', obj_file                  # output object file
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ fallback compilation successful: {obj_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ fallback compilation failed: {e.stderr}")
            print("💡 generated c code is still available in output/model.c")
            raise

    def convert(self):
        """main conversion pipeline"""
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
        
        print("📝 generating header file...")
        header_content = self.generate_header_file()
        with open(os.path.join(self.output_dir, 'model.h'), 'w') as f:
            f.write(header_content)
        
        print("📝 generating source file...")
        source_content = self.generate_source_file()
        with open(os.path.join(self.output_dir, 'model.c'), 'w') as f:
            f.write(source_content)
        
        print("🔨 compiling to arm cortex-m4...")
        self.compile_to_arm_cortex_m4()
        
        print("✅ dynamic conversion complete!")

if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'model.pth'
    converter = DynamicPyToCConverter(model_path)
    converter.convert()