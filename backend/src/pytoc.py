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
        model = torch.load(self.model_path, map_location='cpu')
        model.eval()
        # assume nn.Sequential of Linear layers
        from torch import nn
        for module in model:
            if isinstance(module, nn.Linear):
                w = module.weight.detach().numpy()
                b = module.bias.detach().numpy()
                in_dim, out_dim = w.shape[1], w.shape[0]
                # transpose weight to [out][in]
                self.layers.append((in_dim, out_dim, w.T, b))

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
            externs.append(f'extern const float w{idx}[LAYER{idx}_SIZE][{in_dim}];')
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
        lines = ['#include "model.h"', '', 'static float relu(float x) {', '    return x > 0.0f ? x : 0.0f;', '}']
        # Define weights and biases
        for idx, (in_dim, out_dim, w, b) in enumerate(self.layers):
            w_str = self.format_array(w)
            b_str = self.format_array(b)
            lines.append(f'const float w{idx}[LAYER{idx}_SIZE][{in_dim}] = {w_str};')
            lines.append(f'const float b{idx}[LAYER{idx}_SIZE] = {b_str};\n')
        # predict function
        lines.append('int predict(const float *input) {')
        lines.append('    const float *prev = input;')
        lines.append('    float buf1[1024], buf2[1024];')
        lines.append('    float *cur = buf1, *nxt = buf2;')
        lines.append(f'    int prev_size = {self.layers[0][0]};')
        for idx, (in_dim, out_dim, _, _) in enumerate(self.layers):
            lines.append(f'    // Layer {idx}')
            lines.append(f'    for (int i = 0; i < LAYER{idx}_SIZE; i++) {{')
            lines.append(f'        float acc = b{idx}[i];')
            lines.append(f'        for (int j = 0; j < prev_size; j++) acc += prev[j] * w{idx}[i][j];')
            if idx < len(self.layers) - 1:
                lines.append('        nxt[i] = relu(acc);')
            else:
                lines.append('        nxt[i] = acc;')
            lines.append('    }')
            lines.append('    prev_size = LAYER{0}_SIZE; prev = nxt; float *tmp = cur; cur = nxt; nxt = tmp;'.format(idx))
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
        cmd = [
            'clang',
            '--target=armv7m-none-eabi',
            '-mcpu=cortex-m4',
            '-mthumb',
            '-O3',
            '-c', c_file,
            '-o', obj_file
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ Compiled C to object: {obj_file}")

    def convert(self):
        self.load_and_parse()
        self.write_header()
        self.write_source()
        self.compile_c()
        print('✅ Conversion + LLVM (clang) compilation complete')

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'model.pth'
    PyToCConverter(path).convert()
