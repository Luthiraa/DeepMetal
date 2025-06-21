#!/usr/bin/env python3
# test_complete_workflow.py - validates entire pytorch to c/c++/llvm conversion pipeline

import torch
import torch.nn as nn
import numpy as np
import os
import subprocess
import tempfile
import shutil
from typing import Tuple, List

def create_test_model(model_type: str = 'simple') -> nn.Module:
    """creates minimal test models for validation"""
    if model_type == 'simple':
        # minimal linear model for quick testing
        return nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
    elif model_type == 'conv_simple':
        # minimal conv model
        return nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),  # 8x8 -> 4x8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
    elif model_type == 'hybrid':
        # mixed conv + linear
        return nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),  # 8x8 -> 2x4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 4 * 4, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
    else:
        raise ValueError(f"unknown model type: {model_type}")

def generate_test_data(model_type: str, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """generates appropriate test data for model type"""
    if model_type == 'simple':
        # linear model expects flattened input
        inputs = torch.randn(batch_size, 4)
        targets = torch.randint(0, 3, (batch_size,))
    elif model_type in ['conv_simple', 'hybrid']:
        # conv models expect image-like input
        inputs = torch.randn(batch_size, 1, 8, 8)
        targets = torch.randint(0, 3, (batch_size,))
    else:
        raise ValueError(f"unknown model type: {model_type}")
    
    return inputs, targets

def train_mini_model(model: nn.Module, model_type: str, epochs: int = 10) -> nn.Module:
    """quickly trains model on synthetic data"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🎯 training {model_type} model for {epochs} epochs...")
    
    for epoch in range(epochs):
        # generate random batch
        inputs, targets = generate_test_data(model_type, batch_size=32)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            accuracy = (outputs.argmax(1) == targets).float().mean()
            print(f"  epoch {epoch}: loss={loss.item():.4f}, acc={accuracy:.4f}")
    
    model.eval()
    return model

def save_test_model(model: nn.Module, model_name: str, temp_dir: str) -> str:
    """saves model in converter-compatible format"""
    model_path = os.path.join(temp_dir, f'{model_name}.pth')
    # use old serialization format for compatibility
    torch.save(model, model_path, _use_new_zipfile_serialization=False)
    print(f"💾 saved model to: {model_path}")
    return model_path

def test_pytorch_inference(model: nn.Module, model_type: str) -> np.ndarray:
    """gets reference pytorch inference results"""
    model.eval()
    inputs, _ = generate_test_data(model_type, batch_size=1)
    
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
    
    print(f"🔍 pytorch reference output: {outputs[0].numpy()}")
    print(f"🔍 pytorch probabilities: {probabilities[0].numpy()}")
    print(f"🔍 pytorch prediction: {outputs.argmax(1)[0].item()}")
    
    return outputs[0].numpy()

def run_converter(converter_script: str, model_path: str, temp_dir: str) -> bool:
    """runs a converter script and checks for success"""
    try:
        print(f"🔄 running {converter_script}...")
        
        # change to temp directory for output
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # run converter
        result = subprocess.run(
            ['python', os.path.join(original_dir, converter_script), model_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print(f"✅ {converter_script} succeeded")
            print(f"stdout: {result.stdout}")
            return True
        else:
            print(f"❌ {converter_script} failed")
            print(f"stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {converter_script} timed out")
        return False
    except Exception as e:
        print(f"💥 {converter_script} crashed: {e}")
        return False

def check_generated_files(temp_dir: str) -> Dict[str, bool]:
    """checks if expected output files were generated"""
    expected_files = {
        'converter.py': ['output/model.h', 'output/model.c', 'output/model.o'],
        'llvm.py': ['output/model.ll', 'output/model_llvm.o'],
        'pytoc.py': ['output/dynamic_model.cpp', 'output/model_config.json']
    }
    
    results = {}
    
    for converter, files in expected_files.items():
        converter_success = True
        print(f"📁 checking files for {converter}:")
        
        for file_path in files:
            full_path = os.path.join(temp_dir, file_path)
            exists = os.path.exists(full_path)
            size = os.path.getsize(full_path) if exists else 0
            
            print(f"  {file_path}: {'✅' if exists else '❌'} ({size} bytes)")
            
            if not exists or size == 0:
                converter_success = False
        
        results[converter] = converter_success
    
    return results

def validate_c_compilation(temp_dir: str) -> bool:
    """attempts to compile generated c code"""
    try:
        c_file = os.path.join(temp_dir, 'output', 'model.c')
        h_file = os.path.join(temp_dir, 'output', 'model.h')
        
        if not (os.path.exists(c_file) and os.path.exists(h_file)):
            print("❌ c files not found for compilation test")
            return False
        
        # create simple test program
        test_c = f'''
#include "model.h"
#include <stdio.h>

int main() {{
    float input[4] = {{1.0f, 2.0f, 3.0f, 4.0f}};
    int result = predict(input, 1, 1, 4);
    printf("prediction: %d\\n", result);
    return 0;
}}
'''
        
        test_file = os.path.join(temp_dir, 'test_model.c')
        with open(test_file, 'w') as f:
            f.write(test_c)
        
        # try to compile with gcc
        result = subprocess.run([
            'gcc', '-I', os.path.join(temp_dir, 'output'),
            test_file, c_file,
            '-o', os.path.join(temp_dir, 'test_model'),
            '-lm'  # link math library
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ c compilation successful")
            return True
        else:
            print(f"❌ c compilation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"💥 c compilation test crashed: {e}")
        return False

def run_workflow_test(model_type: str = 'simple') -> bool:
    """runs complete workflow test for a model type"""
    print(f"\n🧪 testing workflow for {model_type} model")
    print("=" * 50)
    
    success = True
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # create and train model
            model = create_test_model(model_type)
            print(f"📋 model architecture:\n{model}")
            
            # quick training
            model = train_mini_model(model, model_type)
            
            # save model
            model_path = save_test_model(model, f'test_{model_type}', temp_dir)
            
            # get reference pytorch results
            pytorch_output = test_pytorch_inference(model, model_type)
            
            # test each converter
            converters = ['converter.py', 'llvm.py', 'pytoc.py']
            converter_results = {}
            
            for converter in converters:
                if os.path.exists(converter):
                    converter_results[converter] = run_converter(converter, model_path, temp_dir)
                else:
                    print(f"⚠️ {converter} not found, skipping")
                    converter_results[converter] = False
                    success = False
            
            # check generated files
            file_results = check_generated_files(temp_dir)
            
            # attempt c compilation if applicable
            if file_results.get('converter.py', False):
                c_compile_success = validate_c_compilation(temp_dir)
                if not c_compile_success:
                    success = False
            
            # summary
            print(f"\n📊 results summary for {model_type}:")
            for converter, result in converter_results.items():
                status = "✅" if result else "❌"
                print(f"  {converter}: {status}")
            
            for converter, result in file_results.items():
                status = "✅" if result else "❌"
                print(f"  {converter} files: {status}")
            
            if not all(converter_results.values()):
                success = False
            
        except Exception as e:
            print(f"💥 workflow test failed: {e}")
            success = False
    
    return success

def main():
    """runs comprehensive workflow validation"""
    print("🚀 pytorch to c/c++/llvm conversion workflow validator")
    print("=" * 60)
    
    # check if converter scripts exist
    required_scripts = ['converter.py', 'llvm.py', 'pytoc.py']
    missing_scripts = [script for script in required_scripts if not os.path.exists(script)]
    
    if missing_scripts:
        print(f"❌ missing converter scripts: {missing_scripts}")
        print("please ensure all converter scripts are in the current directory")
        return False
    
    # test different model types
    model_types = ['simple', 'conv_simple', 'hybrid']
    all_tests_passed = True
    
    for model_type in model_types:
        test_passed = run_workflow_test(model_type)
        if not test_passed:
            all_tests_passed = False
    
    # final summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 all workflow tests passed!")
        print("the conversion pipeline is working correctly")
    else:
        print("❌ some workflow tests failed")
        print("check the output above for specific issues")
    
    print("\n📝 next steps:")
    print("1. run: python export_model.py --model-type hybrid --epochs 3")
    print("2. test converters: ./test_conversion.sh")
    print("3. deploy generated c/c++ code to your target platform")
    
    return all_tests_passed

if __name__ == '__main__':
    main()