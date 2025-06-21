#!/usr/bin/env python3
# analyze_memory.py - analyzes neural network memory usage for stm32f446re

import os
import sys
import re

def read_max_buffer_size():
    """reads actual MAX_BUFFER_SIZE from generated header"""
    model_h = "output/model.h"
    if not os.path.exists(model_h):
        return None
    
    with open(model_h, 'r') as f:
        content = f.read()
    
    # search for #define MAX_BUFFER_SIZE value
    match = re.search(r'#define\s+MAX_BUFFER_SIZE\s+(\d+)', content)
    if match:
        return int(match.group(1))
    return None

def count_weight_arrays():
    """counts actual weight arrays and their sizes from model.c"""
    model_c = "output/model.c"
    if not os.path.exists(model_c):
        return 0, 0
    
    with open(model_c, 'r') as f:
        content = f.read()
    
    # count const float arrays (weights and biases)
    weight_matches = re.findall(r'const\s+float\s+\w+\[[\d\[\]]+\]\s+=\s+{', content)
    total_weight_elements = 0
    
    # rough estimate: count float literals in weight definitions
    # this is approximate but better than 80% heuristic
    float_matches = re.findall(r'[-+]?\d*\.?\d+f', content)
    total_weight_elements = len(float_matches)
    
    return len(weight_matches), total_weight_elements

def analyze_model_memory():
    """analyzes generated model memory requirements"""
    print("🔍 STM32F446RE Memory Analysis")
    print("=" * 40)
    
    # STM32F446RE specifications
    flash_size = 512 * 1024  # 512KB Flash
    ram_size = 128 * 1024    # 128KB RAM
    
    print(f"STM32F446RE Specifications:")
    print(f"  Flash: {flash_size // 1024}KB")
    print(f"  RAM:   {ram_size // 1024}KB")
    print()
    
    # Check if model files exist
    model_c = "output/model.c"
    model_h = "output/model.h"
    
    if not os.path.exists(model_c):
        print("❌ output/model.c not found")
        print("🔧 run: python converter.py models/your_model.pth")
        return
    
    # Estimate model size
    c_size = os.path.getsize(model_c)
    h_size = os.path.getsize(model_h)
    
    print(f"Generated Code Size:")
    print(f"  model.c: {c_size // 1024}KB ({c_size} bytes)")
    print(f"  model.h: {h_size // 1024}KB ({h_size} bytes)")
    print()
    
    # Analyze actual weight data
    num_arrays, total_weights = count_weight_arrays()
    weight_data_size = total_weights * 4  # 4 bytes per float
    code_size = c_size - weight_data_size  # approximate code size
    
    print(f"Detailed Flash Analysis:")
    print(f"  Weight arrays: {num_arrays}")
    print(f"  Total weights: {total_weights:,}")
    print(f"  Weight data: ~{weight_data_size // 1024}KB ({weight_data_size} bytes)")
    print(f"  Code size: ~{code_size // 1024}KB ({code_size} bytes)")
    
    flash_percent = (c_size / flash_size) * 100
    print(f"  Total flash usage: ~{c_size // 1024}KB ({flash_percent:.1f}%)")
    print()
    
    # Read actual buffer size from header
    buffer_size = read_max_buffer_size()
    if buffer_size:
        print(f"Actual RAM Analysis (from model.h):")
        print(f"  MAX_BUFFER_SIZE: {buffer_size} floats")
        
        # two buffers used in generated C code
        buffer_ram = buffer_size * 4 * 2  # 2 buffers * 4 bytes per float
        stack_estimate = 2048  # rough estimate for stack, local vars
        total_ram = buffer_ram + stack_estimate
        
        ram_percent = (total_ram / ram_size) * 100
        
        print(f"  Buffer memory: {buffer_ram // 1024}KB ({buffer_size} * 4 bytes * 2 buffers)")
        print(f"  Stack estimate: {stack_estimate // 1024}KB")
        print(f"  Total RAM: ~{total_ram // 1024}KB ({ram_percent:.1f}%)")
    else:
        print("⚠️ Could not read MAX_BUFFER_SIZE from model.h")
        print("Using fallback RAM analysis...")
        
        # fallback to old method but with corrected values
        buffer_ram = 8192 * 4 * 2  # default buffer size * 2 buffers
        ram_percent = (buffer_ram / ram_size) * 100
        print(f"  Estimated RAM: ~{buffer_ram // 1024}KB ({ram_percent:.1f}%)")
    
    print()
    
    # Enhanced recommendations
    print("📊 Memory Recommendations:")
    if flash_percent > 90:
        print("🚨 Flash usage critical! Model too large for STM32F446RE")
    elif flash_percent > 80:
        print("⚠️ Flash usage high! Consider:")
        print("   - Smaller model (--model-size tiny)")
        print("   - Weight quantization (float32 → int8)")
        print("   - Layer pruning")
    elif flash_percent > 50:
        print("💡 Flash usage moderate - room for optimization")
    else:
        print("✅ Flash usage acceptable")
    
    if buffer_size and ram_percent > 90:
        print("🚨 RAM usage critical! Reduce MAX_BUFFER_SIZE")
    elif buffer_size and ram_percent > 80:
        print("⚠️ RAM usage high! Consider:")
        print(f"   - Reduce buffer size in converter.py: max_buffer_size = {buffer_size // 2}")
        print("   - Use smaller model architecture")
    elif buffer_size and ram_percent > 50:
        print("💡 RAM usage moderate")
    else:
        print("✅ RAM usage acceptable")
    
    # Buffer size recommendations
    if buffer_size:
        print(f"\n🔧 Buffer Size Analysis:")
        print(f"  Current: {buffer_size} floats ({buffer_size * 4} bytes)")
        
        # calculate minimum needed buffer for this model
        # this would require parsing the actual layer sizes, but we can estimate
        if total_weights > 0:
            # rough estimate: largest layer is probably linear layer
            # find largest layer from weight count patterns
            estimated_min_buffer = min(buffer_size, max(1024, total_weights // 10))
            print(f"  Estimated minimum: ~{estimated_min_buffer} floats")
            
            if buffer_size > estimated_min_buffer * 2:
                print(f"  💡 Consider reducing to: {estimated_min_buffer * 2}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--detailed':
        # show detailed file analysis
        model_c = "output/model.c"
        if os.path.exists(model_c):
            print("📄 Detailed File Analysis:")
            with open(model_c, 'r') as f:
                lines = f.readlines()
            
            weight_lines = [i for i, line in enumerate(lines) if 'const float' in line and '=' in line]
            print(f"  Weight definitions start at lines: {weight_lines[:5]}...")
            print(f"  Total lines in model.c: {len(lines)}")
            print()
    
    analyze_model_memory()

if __name__ == '__main__':
    main()