import torch
import numpy as np
from llvmlite import ir

class EdgeAICompiler:
    def __init__(self, model_path, input_size, layer_sizes, activation='relu'):
        self.model_path = model_path
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = {}

    def load_weights(self):
        model = torch.load(self.model_path, map_location='cpu')
        model.eval()
        for name, param in model.named_parameters():
            self.weights[name] = param.detach().numpy()

    def build_llvm_ir(self):
        module = ir.Module(name="edge_model")
        float_type = ir.FloatType()

        input_type = ir.ArrayType(float_type, self.input_size)
        func_type = ir.FunctionType(float_type, [input_type.as_pointer()])
        predict_fn = ir.Function(module, func_type, name="predict")

        block = predict_fn.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        input_ptr = predict_fn.args[0]
        prev_output = [builder.load(builder.gep(input_ptr, [ir.Constant(ir.IntType(32), 0),
                                                            ir.Constant(ir.IntType(32), i)]))
                       for i in range(self.input_size)]

        weight_index = 0
        for layer_num, (in_dim, out_dim) in enumerate(zip([self.input_size] + self.layer_sizes[:-1], self.layer_sizes)):
            next_output = []
            for i in range(out_dim):
                acc = ir.Constant(float_type, self.weights[f'{weight_index}.bias'][i])
                for j in range(in_dim):
                    w = ir.Constant(float_type, self.weights[f'{weight_index}.weight'][i][j])
                    acc = builder.fadd(acc, builder.fmul(prev_output[j], w))
                
                if self.activation == 'relu' and layer_num < len(self.layer_sizes) - 1:
                    acc = builder.select(
                        builder.fcmp_ordered('>=', acc, ir.Constant(float_type, 0.0)),
                        acc,
                        ir.Constant(float_type, 0.0)
                    )
                next_output.append(acc)
            prev_output = next_output
            weight_index += 2

        max_val = prev_output[0]
        max_idx = ir.Constant(ir.IntType(32), 0)
        for i in range(1, len(prev_output)):
            is_bigger = builder.fcmp_ordered('>', prev_output[i], max_val)
            max_val = builder.select(is_bigger, prev_output[i], max_val)
            max_idx = builder.select(is_bigger, ir.Constant(ir.IntType(32), i), max_idx)

        ret = builder.uitofp(max_idx, float_type)
        builder.ret(ret)

        return module

    def compile(self, output_path='output/model.ll'):
        self.load_weights()
        module = self.build_llvm_ir()
        with open(output_path, 'w') as f:
            f.write(str(module))
        print(f"✅ LLVM IR written to {output_path}")