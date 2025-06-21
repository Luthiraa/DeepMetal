#!/usr/bin/env python3
# analyze_memory.py - analyzes neural network memory usage for stm32f446re

import os
import sys

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
    
    # Rough flash usage estimate (weights + code)
    estimated_flash = c_size * 0.8  # weights take ~80% of .c file
    flash_percent = (estimated_flash / flash_size) * 100
    
    print(f"Estimated Memory Usage:")
    print(f"  Flash (weights): ~{estimated_flash // 1024}KB ({flash_percent:.1f}%)")
    
    # RAM usage estimate (buffers + stack)
    max_buffer_size = 65536 * 4  # MAX_BUFFER_SIZE * sizeof(float)
    ram_percent = (max_buffer_size / ram_size) * 100
    
    print(f"  RAM (buffers):   ~{max_buffer_size // 1024}KB ({ram_percent:.1f}%)")
    print()
    
    # Recommendations
    if flash_percent > 80:
        print("⚠️  Flash usage high! Consider model compression")
    elif flash_percent > 50:
        print("💡 Flash usage moderate - consider optimization")
    else:
        print("✅ Flash usage acceptable")
    
    if ram_percent > 80:
        print("⚠️  RAM usage high! Reduce MAX_BUFFER_SIZE")
    elif ram_percent > 50:
        print("💡 RAM usage moderate")
    else:
        print("✅ RAM usage acceptable")

if __name__ == '__main__':
    analyze_model_memory()
