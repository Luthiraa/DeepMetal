# DEEPMETAL

*Transforming AI into Seamless Embedded Powerhouse*
*Compiler for high-level ML libraries to run your models on edge*

![Last Commit](https://img.shields.io/badge/last%20commit-last%20sunday-6f42c1)
![LLVM](https://img.shields.io/badge/llvm-32.0%25-blue)
![Languages](https://img.shields.io/badge/languages-9-blue)

---

### *Built with the tools and technologies:*

![JSON](https://img.shields.io/badge/-JSON-black?logo=json)
![Markdown](https://img.shields.io/badge/-Markdown-000000?logo=markdown)
![npm](https://img.shields.io/badge/-npm-CB3837?logo=npm)
![Dart](https://img.shields.io/badge/-Dart-0175C2?logo=dart)
![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?logo=javascript)
![SymPy](https://img.shields.io/badge/-SymPy-8bc34a?logo=sympy)
![GNU Bash](https://img.shields.io/badge/-GNU%20Bash-4EAA25?logo=gnubash)
![React](https://img.shields.io/badge/-React-61DAFB?logo=react)

![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy)
![C++](https://img.shields.io/badge/-C++-00599C?logo=cplusplus)
![Python](https://img.shields.io/badge/-Python-3776AB?logo=python)
![C](https://img.shields.io/badge/-C-A8B9CC?logo=c)
![ONNX](https://img.shields.io/badge/-ONNX-005CED?logo=onnx)
![Vite](https://img.shields.io/badge/-Vite-646CFF?logo=vite)
![ESLint](https://img.shields.io/badge/-ESLint-4B32C3?logo=eslint)

Complete pipeline for converting PyTorch neural networks to optimized C, C++, and LLVM code for deployment on embedded systems.

## Quick Start

```bash
# 1. create and train a model
python export_model.py --model-type hybrid --epochs 5

# 2. test all converters
./test_conversion.sh

# 3. validate entire workflow
python test_complete_workflow.py
```

## Components

### 1. Model Export (`export_model.py`)
Creates PyTorch models compatible with the conversion pipeline.

**Supported architectures:**
- `linear`: Fully connected layers only (784→128→64→10)
- `conv`: Convolutional layers + linear classifier
- `hybrid`: Mixed conv + linear layers (recommended)

**Usage:**
```bash
# train a hybrid model for 5 epochs
python export_model.py --model-type hybrid --epochs 5 --batch-size 64

# create model without training (for testing)
python export_model.py --model-type linear --no-train

# train on gpu if available
python export_model.py --model-type conv --device cuda --epochs 10
```

**Output:**
- `models/mnist_hybrid_model.pth` - Complete model
- `models/mnist_hybrid_model_state_dict.pth` - State dict only
- `test_conversion.sh` - Script to test all converters

### 2. Dynamic C Converter (`converter.py`)
Generates pure C code optimized for ARM Cortex-M4 microcontrollers.

**Features:**
- Static memory allocation
- Ping-pong buffer optimization
- ARM Cortex-M4 compilation
- Minimal dependencies

**Usage:**
```bash
python converter.py models/mnist_hybrid_model.pth
```

**Output:**
```
output/
├── model.h          # header with declarations
├── model.c          # implementation
└── model.o          # compiled ARM object file
```

**Generated API:**
```c
int predict(const float *input, int input_h, int input_w, int input_ch);
```

### 3. LLVM IR Converter (`llvm.py`)
Generates LLVM intermediate representation with advanced optimizations.

**Features:**
- Cross-platform target support
- Advanced optimization passes
- Loop unrolling and vectorization
- Multiple architecture support

**Usage:**
```bash
python llvm.py models/mnist_hybrid_model.pth
```

**Output:**
```
output/
├── model.ll         # llvm ir code
└── model_llvm.o     # optimized object file
```

### 4. C++ Template Converter (`pytoc.py`)
Generates modern C++ code with STL containers and type safety.

**Features:**
- Template-based architecture
- STL containers for safety
- Easy debugging and modification
- JSON configuration export

**Usage:**
```bash
python pytoc.py models/mnist_hybrid_model.pth
```

**Output:**
```
output/
├── dynamic_model.cpp     # complete c++ implementation
├── dynamic_model         # compiled executable
└── model_config.json     # architecture metadata
```

## Supported Layer Types

### Linear Layers (`torch.nn.Linear`)
```python
nn.Linear(in_features, out_features, bias=True)
```
- Fully connected transformation
- Optional bias terms
- Efficient matrix multiplication

### Convolutional Layers (`torch.nn.Conv2d`)
```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```
- 2D convolution with configurable parameters
- Stride and padding support
- Boundary condition handling

### ReLU Activation (`torch.nn.ReLU`)
```python
nn.ReLU()
```
- Element-wise max(0, x) operation
- Hardware-optimized implementation

## Memory Layout and Optimization

### Data Organization
```c
// weights stored as [output_neurons][input_features]
const float w0[128][784] = {...};

// ping-pong buffers for layer outputs
float buf1[MAX_BUFFER_SIZE], buf2[MAX_BUFFER_SIZE];
```

### Convolution Memory Access
```c
// input indexed as [channel][height][width]
int input_idx = ic * input_h * input_w + ih * input_w + iw;

// output organized as [channel][height][width] 
int output_idx = oc * out_h * out_w + oh * out_w + ow;
```

### Buffer Management
- **Ping-pong buffers**: Alternate between `buf1` and `buf2` for each layer
- **Static allocation**: No dynamic memory allocation for embedded safety
- **Size optimization**: Reuse buffers across layers

## Target Platform Configuration

### ARM Cortex-M4 (Default)
```bash
clang --target=armv7em-none-eabi \
      -mcpu=cortex-m4 \
      -mthumb \
      -mfloat-abi=hard \
      -mfpu=fpv4-sp-d16 \
      -O3
```

### Custom Targets
Modify compiler flags in each converter for different targets:
- **x86**: `--target=x86_64-linux-gnu`
- **ARM64**: `--target=aarch64-linux-gnu`  
- **RISC-V**: `--target=riscv32-unknown-elf`

## Performance Characteristics

### Model Size vs. Accuracy Trade-offs
```
Linear (784→128→64→10):  ~107K parameters, ~95% accuracy
Conv (1×3×3→16×3×3→32): ~23K parameters, ~98% accuracy  
Hybrid (optimized):      ~15K parameters, ~97% accuracy
```

### Memory Requirements
```
Flash (weights):  15KB - 400KB depending on architecture
RAM (buffers):    2KB - 64KB for intermediate computations
Stack:           <1KB for local variables
```

### Inference Speed (Cortex-M4 @ 80MHz)
```
Linear model:    ~2ms per inference
Conv model:      ~8ms per inference
Hybrid model:    ~5ms per inference
```

## Advanced Usage

### Custom Model Architecture
```python
# create your own sequential model
custom_model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, stride=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(32 * 14 * 14, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# export for conversion
torch.save(custom_model, 'custom_model.pth')
```

### Integration with Embedded Systems
```c
// example main.c for microcontroller
#include "model.h"
#include <stdio.h>

float sensor_data[784];  // input from sensors

int main() {
    // collect sensor data
    read_sensors(sensor_data);
    
    // run inference  
    int prediction = predict(sensor_data, 28, 28, 1);
    
    // act on prediction
    handle_prediction(prediction);
    
    return 0;
}
```

### Debugging and Validation
```bash
# validate complete workflow
python test_complete_workflow.py

# compare outputs between pytorch and c
python validate_conversion.py model.pth

# profile performance
python benchmark_inference.py model.pth
```

## Troubleshooting

### Common Issues

**1. Unsupported layer types**
```
Error: Layer type 'BatchNorm2d' not supported
```
*Solution*: Remove or replace unsupported layers. Currently supported: Linear, Conv2d, ReLU.

**2. Memory buffer overflow**
```
Error: Layer output size exceeds buffer capacity
```
*Solution*: Increase `MAX_BUFFER_SIZE` in converter or reduce model size.

**3. LLVM compilation failures**
```
Error: llvmlite not installed
```
*Solution*: `pip install llvmlite` or use C/C++ converters instead.

**4. ARM toolchain missing**
```
Error: clang: command not found
```
*Solution*: Install ARM GCC toolchain or use x86 targets for testing.

### Model Architecture Guidelines

**For embedded deployment:**
- Keep total parameters under 100K
- Avoid large convolutional layers
- Use stride > 1 to reduce spatial dimensions quickly
- Prefer ReLU over other activations

**For best converter compatibility:**
- Use `nn.Sequential` models
- Avoid custom layers or complex control flow
- Keep all operations differentiable
- Save complete models, not just state dicts

## Dependencies

**Python packages:**
```bash
pip install torch torchvision numpy llvmlite
```

**System tools:**
```bash
# ubuntu/debian
sudo apt install clang gcc-arm-none-eabi

# macos
brew install llvm arm-none-eabi-gcc

# or use conda
conda install pytorch torchvision llvmlite
```

## Next Steps

1. **Create your model**: `python export_model.py --model-type hybrid`
2. **Test conversion**: `./test_conversion.sh`
3. **Integrate with firmware**: Use generated `.o` files in your embedded project
4. **Optimize further**: Profile and adjust model architecture for your constraints
5. **Deploy**: Flash to target hardware and validate real-world performance

For more advanced use cases, see the individual converter documentation and consider extending the layer support for your specific requirements.
