#!/usr/bin/env python3
"""
Test script to demonstrate the proper DeepMetal workflow
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add backend src to path
sys.path.append(os.path.join('backend', 'src'))

def test_proper_workflow():
    """Test the proper workflow: Model -> Inference -> Converter -> STM32 Code"""
    
    print("🧪 Testing Proper DeepMetal Workflow")
    print("=" * 50)
    
    # Step 1: Load or create PyTorch model
    print("\n1️⃣ Loading PyTorch Model...")
    try:
        from export_model import create_sequential_model
        
        # Create a small MNIST model
        model = create_sequential_model(model_type='linear', model_size='tiny')
        model.eval()
        print(f"✓ Model created: {model}")
        print(f"✓ Parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Save model for converter
        model_path = 'test_model.pth'
        torch.save(model, model_path)
        print(f"✓ Model saved to: {model_path}")
        
    except ImportError as e:
        print(f"⚠ Could not import export_model: {e}")
        # Create simple fallback model
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )
        model.eval()
        torch.save(model, model_path)
        print(f"✓ Fallback model created and saved")
    
    # Step 2: Create test image (simulate uploaded image)
    print("\n2️⃣ Creating Test Image...")
    # Create a simple test pattern (simulating digit '3')
    test_image = np.zeros((28, 28), dtype=np.float32)
    # Add some pattern to simulate a digit
    test_image[5:23, 8:20] = 0.8  # Simple rectangle pattern
    test_image[10:18, 10:18] = 0.2  # Inner area
    
    # Normalize like MNIST
    test_image = (test_image - 0.1307) / 0.3081
    print(f"✓ Test image created: {test_image.shape}")
    print(f"✓ Image range: [{test_image.min():.3f}, {test_image.max():.3f}]")
    
    # Step 3: Run PyTorch inference
    print("\n3️⃣ Running PyTorch Inference...")
    model.eval()
    with torch.no_grad():
        # Flatten the image for linear model input
        input_tensor = torch.tensor(test_image, dtype=torch.float32).flatten().unsqueeze(0)
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        print(f"✓ Raw output: {output.numpy()}")
        print(f"✓ Prediction: {prediction}")
        print(f"✓ Confidence: {confidence:.3f}")
        print(f"✓ All probabilities: {probabilities.numpy()}")
    
    # Step 4: Generate STM32 C code using converter
    print("\n4️⃣ Generating STM32 C Code...")
    try:
        from converter import DynamicPyToCConverter
        
        output_dir = 'test_output'
        converter = DynamicPyToCConverter(model_path, output_dir)
        
        # Parse model architecture
        converter.parse_model_architecture()
        print(f"✓ Model parsed: {len(converter.layer_configs)} layers")
        
        # Generate C code
        converter.convert()
        print(f"✓ C code generated in: {output_dir}")
        
        # Read generated files
        model_c_path = os.path.join(output_dir, 'model.c')
        model_h_path = os.path.join(output_dir, 'model.h')
        
        if os.path.exists(model_c_path):
            with open(model_c_path, 'r') as f:
                model_c_content = f.read()
            print(f"✓ model.c size: {len(model_c_content)} characters")
        
        if os.path.exists(model_h_path):
            with open(model_h_path, 'r') as f:
                model_h_content = f.read()
            print(f"✓ model.h size: {len(model_h_content)} characters")
        
    except ImportError as e:
        print(f"⚠ Could not import converter: {e}")
        print("⚠ Skipping C code generation")
        model_c_content = "// Converter not available"
        model_h_content = "// Converter not available"
    
    # Step 5: Create STM32 main.c with actual image data
    print("\n5️⃣ Creating STM32 Main File...")
    
    # Convert image to C array format
    image_c_array = ', '.join([f'{x:.6f}f' for x in test_image.flatten()])
    
    main_c_content = f'''// main.c - STM32 MNIST Neural Network
// Generated from proper workflow test
// Predicted digit: {prediction} (confidence: {confidence:.3f})

#include "model.h"
#include <stdint.h>

// Input image data (28x28 MNIST format)
static const float input_image[784] = {{ {image_c_array} }};

// STM32 hardware abstraction
void setup_gpio() {{
    // Enable GPIOA clock
    *((volatile unsigned int*)0x40023830) |= 1;
    
    // Configure PA5 as output (LED)
    volatile unsigned int *moder = (volatile unsigned int*)0x40020000;
    *moder = (*moder & ~(0x3 << 10)) | (1 << 10);
}}

void led_on() {{ *((volatile unsigned int*)0x40020018) = (1 << 5); }}
void led_off() {{ *((volatile unsigned int*)0x40020018) = (1 << 21); }}

void delay_ms(int ms) {{
    for(volatile int i = 0; i < ms * 1000; i++);
}}

void led_blink(int count) {{
    for(int i = 0; i < count; i++) {{
        led_on(); delay_ms(250); led_off(); delay_ms(250);
    }}
}}

int main() {{
    setup_gpio();
    
    // Initial startup blink
    for(int i = 0; i < 3; i++) {{
        led_on(); delay_ms(100); led_off(); delay_ms(100);
    }}
    delay_ms(1000);
    
    while(1) {{
        // Run neural network inference
        int result = predict(input_image, 28, 28, 1);
        
        // Blink LED based on prediction (result + 1 for digit 0)
        led_blink(result + 1);
        delay_ms(3000);
    }}
    
    return 0;
}}

// Reset handler for STM32
__attribute__((section(".isr_vector")))
unsigned int vector_table[] = {{ 0x20020000, (unsigned int)main }};

__attribute__((naked)) void Reset_Handler() {{
    main();
    while(1);
}}
'''
    
    # Save main.c
    main_c_path = os.path.join(output_dir, 'main.c')
    os.makedirs(output_dir, exist_ok=True)
    with open(main_c_path, 'w') as f:
        f.write(main_c_content)
    
    print(f"✓ main.c created: {len(main_c_content)} characters")
    print(f"✓ Files saved in: {output_dir}")
    
    # Step 6: Test compilation (if ARM GCC available)
    print("\n6️⃣ Testing Compilation...")
    try:
        import subprocess
        
        # Check if ARM GCC is available
        result = subprocess.run(['arm-none-eabi-gcc', '--version'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ ARM GCC found, attempting compilation...")
            
            # Compile model.c
            cmd = [
                'arm-none-eabi-gcc',
                '-mcpu=cortex-m4',
                '-mthumb',
                '-mfloat-abi=hard',
                '-mfpu=fpv4-sp-d16',
                '-O2',
                '-c',
                model_c_path,
                '-o', os.path.join(output_dir, 'model.o')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
            
            if result.returncode == 0:
                print("✓ Model compilation successful")
            else:
                print(f"⚠ Model compilation failed: {result.stderr}")
            
            # Compile main.c
            cmd = [
                'arm-none-eabi-gcc',
                '-mcpu=cortex-m4',
                '-mthumb',
                '-mfloat-abi=hard',
                '-mfpu=fpv4-sp-d16',
                '-O2',
                '-c',
                main_c_path,
                '-o', os.path.join(output_dir, 'main.o')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
            
            if result.returncode == 0:
                print("✓ Main compilation successful")
            else:
                print(f"⚠ Main compilation failed: {result.stderr}")
                
        else:
            print("⚠ ARM GCC not found - skipping compilation")
            print("  Install with: sudo apt-get install gcc-arm-none-eabi")
    
    except Exception as e:
        print(f"⚠ Compilation test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Proper Workflow Test Complete!")
    print(f"📊 Results:")
    print(f"   • PyTorch prediction: {prediction}")
    print(f"   • Confidence: {confidence:.3f}")
    print(f"   • Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"   • Generated files: {output_dir}/")
    print(f"   • C code generated: {'✓' if 'converter' in locals() else '⚠'}")
    print(f"   • Compilation tested: {'✓' if 'result' in locals() else '⚠'}")
    
    print(f"\n📁 Generated files in {output_dir}:")
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            size = os.path.getsize(file_path)
            print(f"   • {file} ({size} bytes)")
    
    print(f"\n🔧 Next steps:")
    print(f"   1. Flash the generated .o files to STM32")
    print(f"   2. The LED will blink {prediction + 1} times")
    print(f"   3. Real neural network inference on microcontroller!")

if __name__ == '__main__':
    test_proper_workflow() 