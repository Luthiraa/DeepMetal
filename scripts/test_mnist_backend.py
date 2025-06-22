#!/usr/bin/env python3
"""
Test script for MNIST Flask Backend with Model Export
"""

import requests
import json
import time
import numpy as np
from PIL import Image
import io

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get('http://localhost:5000/api/health')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n🔍 Testing model info...")
    try:
        response = requests.get('http://localhost:5000/api/model-info')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info: {data}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def create_test_image():
    """Create a simple test MNIST-like image"""
    # Create a 28x28 grayscale image with a simple pattern
    img_array = np.zeros((28, 28), dtype=np.uint8)
    
    # Draw a simple "1" pattern
    for i in range(8, 20):
        img_array[i, 14] = 255  # Vertical line
    
    # Add some noise
    noise = np.random.randint(0, 50, (28, 28), dtype=np.uint8)
    img_array = np.clip(img_array + noise, 0, 255)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array, mode='L')
    return img

def test_image_processing():
    """Test image processing with a generated test image"""
    print("\n🔍 Testing image processing...")
    
    try:
        # Create test image
        test_img = create_test_image()
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Prepare the request
        files = {'image': ('test_digit.png', img_bytes, 'image/png')}
        
        print("📤 Sending test image...")
        response = requests.post('http://localhost:5000/api/process-mnist', files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Image processing successful!")
            print(f"   Prediction: {data.get('prediction', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 'N/A'):.4f}")
            print(f"   Processing time: {data.get('processing_time', 'N/A'):.2f}ms")
            print(f"   Model type: {data.get('model_type', 'N/A')}")
            
            # Check if C code was generated
            generated_code = data.get('generated_code', '')
            if generated_code:
                print(f"   C code generated: {len(generated_code)} characters")
                print(f"   Code contains STM32 setup: {'setup_gpio' in generated_code}")
                print(f"   Code contains neural network: {'predict' in generated_code}")
            else:
                print("   ❌ No C code generated")
            
            # Check compilation result
            compilation = data.get('compilation_result', {})
            if compilation:
                print(f"   Compilation: {compilation.get('compilation', 'unknown')}")
                print(f"   Message: {compilation.get('message', 'N/A')}")
            
            return True
        else:
            print(f"❌ Image processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing MNIST Flask Backend with Model Export")
    print("=" * 50)
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Run tests
    tests = [
        test_health_check,
        test_model_info,
        test_image_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the backend logs for details.")

if __name__ == '__main__':
    main() 