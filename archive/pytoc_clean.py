# converter.py
import torch
import numpy as np
import os
import subprocess

class PyToCConverter:
    def __init__(self, model_path, output_dir='output'):
        self.model_path = model_path
        self.output_dir = output_dir
        self.layers = []  # list of (in_dim, out_dim, weight, bias)
        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_parse(self):
        # Load the state_dict (weights dictionary)
        state_dict = torch.load(self.model_path, map_location='cpu')
        
        # Handle state_dict (dictionary of weights)
        # Sort keys to process layers in order
        weight_keys = [k for k in state_dict.keys() if k.endswith('.weight')]
        weight_keys.sort()
        
        print("DEBUG: Found weight keys:", weight_keys)
        
        for weight_key in weight_keys:
            bias_key = weight_key.replace('.weight', '.bias')
            
            if bias_key in state_dict:
                w = state_dict[weight_key].detach().numpy()
                b = state_dict[bias_key].detach().numpy()
                
                print(f"DEBUG: {weight_key} shape: {w.shape}, {bias_key} shape: {b.shape}")
                  # Handle different layer types
                if len(w.shape) == 4:  # Conv2D: (out_channels, in_channels, kernel_h, kernel_w)
                    # Flatten conv weights for now - this is a simplified approach
                    w_flat = w.reshape(w.shape[0], -1)  # (out_channels, in_channels * kernel_h * kernel_w)
                    in_dim, out_dim = w_flat.shape[1], w_flat.shape[0]
                    # Transpose so we have [in_dim][out_dim] for C array access
                    self.layers.append((in_dim, out_dim, w_flat, b))
                elif len(w.shape) == 2:  # Linear: (out_features, in_features)
                    in_dim, out_dim = w.shape[1], w.shape[0]
                    # Transpose so we have [in_dim][out_dim] for C array access
                    self.layers.append((in_dim, out_dim, w, b))
                else:
                    print(f"WARNING: Unsupported weight shape {w.shape} for {weight_key}")

    def format_array(self, arr):
        if arr.ndim == 1:
            return '{' + ', '.join(f'{float(x):.6f}f' for x in arr) + '}'
        elif arr.ndim == 2:
            rows = []
            for row in arr:
                rows.append(self.format_array(row))
            return '{ ' + ', '.join(rows) + ' }'
        else:
            raise ValueError('Unsupported array dimensions')

    def write_header(self):
        # Macro definitions
        macro_lines = []
        for idx, (_, out_dim, _, _) in enumerate(self.layers):
            macro_lines.append(f'#define LAYER{idx}_SIZE {out_dim}')
        macros = '\n'.join(macro_lines)
        externs = []
        for idx, (in_dim, _, _, _) in enumerate(self.layers):
            externs.append(f'extern const float w{idx}[{in_dim}][LAYER{idx}_SIZE];')
            externs.append(f'extern const float b{idx}[LAYER{idx}_SIZE];')
        extern_block = '\n'.join(externs)

        header = f'''#ifndef MODEL_H
#define MODEL_H

{macros}
#define NUM_LAYERS {len(self.layers)}

{extern_block}

int predict(const float *input);
#endif  // MODEL_H
'''
        with open(os.path.join(self.output_dir, 'model.h'), 'w') as f:
            f.write(header)

    def write_source(self):
        lines = ['#include "model.h"', '', 'static float relu(float x) {', '    return x > 0.0f ? x : 0.0f;', '}']        # Define weights and biases
        for idx, (in_dim, out_dim, w, b) in enumerate(self.layers):
            w_str = self.format_array(w)
            b_str = self.format_array(b)
            lines.append(f'const float w{idx}[{in_dim}][LAYER{idx}_SIZE] = {w_str};')
            lines.append(f'const float b{idx}[LAYER{idx}_SIZE] = {b_str};')
        lines.append('')        # predict function
        lines.append('int predict(const float *input) {')
        lines.append('    const float *prev = input;')
        lines.append('    float buf1[1024], buf2[1024];')
        lines.append('    float *cur = buf1, *nxt = buf2;')
        lines.append(f'    int prev_size = {self.layers[0][0]};')
        for idx, (in_dim, out_dim, _, _) in enumerate(self.layers):
            lines.append(f'    // Layer {idx}')
            lines.append(f'    for (int i = 0; i < LAYER{idx}_SIZE; i++) {{')
            lines.append(f'        float acc = b{idx}[i];')
            lines.append(f'        for (int j = 0; j < prev_size; j++) acc += prev[j] * w{idx}[j][i];')
            if idx < len(self.layers) - 1:
                lines.append('        nxt[i] = relu(acc);')
            else:
                lines.append('        nxt[i] = acc;')
            lines.append('    }')
            lines.append(f'    prev_size = LAYER{idx}_SIZE; prev = nxt;')
            lines.append(f'    {{ float *tmp = cur; cur = nxt; nxt = tmp; }}')
        lines.append('    // Argmax')
        lines.append('    int max_i = 0; float max_v = prev[0];')
        lines.append('    for (int i = 1; i < prev_size; i++) { if (prev[i] > max_v) { max_v = prev[i]; max_i = i; } }')
        lines.append('    return max_i;')
        lines.append('}')

        src = '\n'.join(lines)
        with open(os.path.join(self.output_dir, 'model.c'), 'w') as f:
            f.write(src)

    def compile_c(self):
        c_file = os.path.join(self.output_dir, 'model.c')
        obj_file = os.path.join(self.output_dir, 'model.o')
        
        # Use Clang with ARM Cortex-M4 target for embedded systems
        cmd = [
            'clang',
            '--target=armv7m-none-eabi',
            '-mcpu=cortex-m4',
            '-mfloat-abi=hard',
            '-mfpu=fpv4-sp-d16',
            '-O2',
            '-c',
            c_file,
            '-o',
            obj_file
        ]
        
        try:
            print(f"🔨 Compiling with Clang for ARM Cortex-M4...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Successfully compiled with Clang to {obj_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Clang compilation failed: {e}")
            print(f"   stderr: {e.stderr}")
            print(f"   stdout: {e.stdout}")
        except FileNotFoundError:
            print("❌ Clang not found in PATH. Please ensure LLVM is properly installed.")

    def convert(self):
        print(f"🔄 Loading model from {self.model_path}")
        self.load_and_parse()
        print(f"📊 Found {len(self.layers)} layers")
        for i, (in_dim, out_dim, _, _) in enumerate(self.layers):
            print(f"   Layer {i}: {in_dim} -> {out_dim}")
        
        print("📝 Writing header file...")
        self.write_header()
        print("📝 Writing source file...")
        self.write_source()
        print("🔨 Compiling...")
        self.compile_c()
        print("✅ Conversion complete!")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pytoc.py <model.pth>")
        sys.exit(1)
    path = sys.argv[1]
    PyToCConverter(path).convert()
