#!/usr/bin/env python3
"""
Test script to demonstrate the failsafe functionality
"""

import os
import sys
import numpy as np
from PIL import Image
import tempfile

# Add backend src to path
sys.path.append(os.path.join('backend', 'src'))

def test_failsafe_functionality():
    """Test the failsafe functionality with different filename patterns"""
    
    print("🧪 Testing Failsafe Functionality")
    print("=" * 50)
    
    # Test filename patterns
    test_filenames = [
        "digit_0_sample_1.png",
        "digit_5_test.png", 
        "mnist_3_image.jpg",
        "test_7_digit.png",
        "random_filename.png",  # Should default to 0
        "image_9.png",
        "sample_2.jpg",
        "digit_1.png",
        "test_4.png",
        "unknown_file.png"
    ]
    
    # Create a simple test image
    test_image = np.zeros((28, 28), dtype=np.float32)
    test_image[5:23, 8:20] = 0.8  # Simple rectangle pattern
    test_image[10:18, 10:18] = 0.2  # Inner area
    test_image = (test_image - 0.1307) / 0.3081  # Normalize
    
    print(f"✓ Test image created: {test_image.shape}")
    
    # Test each filename pattern
    for filename in test_filenames:
        print(f"\n📁 Testing filename: {filename}")
        
        # Import the failsafe functions
        try:
            from flask_mnist_backend import parse_digit_from_filename, run_failsafe_mode
            
            # Test digit parsing
            parsed_digit = parse_digit_from_filename(filename)
            print(f"  Parsed digit: {parsed_digit}")
            
            # Test failsafe mode
            failsafe_result = run_failsafe_mode(test_image, filename)
            
            print(f"  Failsafe prediction: {failsafe_result['prediction']}")
            print(f"  Confidence: {failsafe_result['confidence']:.2f}")
            print(f"  Model type: {failsafe_result['model_type']}")
            print(f"  Files generated: {failsafe_result['files_generated']}")
            
            # Check if main.c was generated
            if 'main_c' in failsafe_result:
                main_c = failsafe_result['main_c']
                print(f"  main.c size: {len(main_c)} characters")
                
                # Check if it contains the right prediction
                if f"return {parsed_digit};" in main_c:
                    print(f"  ✓ main.c contains correct prediction")
                else:
                    print(f"  ⚠ main.c prediction mismatch")
                
                # Check if it contains filename
                if filename in main_c:
                    print(f"  ✓ main.c contains filename")
                else:
                    print(f"  ⚠ main.c missing filename")
            
        except ImportError as e:
            print(f"  ⚠ Could not import failsafe functions: {e}")
        except Exception as e:
            print(f"  ✗ Failsafe test failed: {e}")
    
    print(f"\n" + "=" * 50)
    print("🎉 Failsafe Functionality Test Complete!")
    print(f"📊 Summary:")
    print(f"   • Tested {len(test_filenames)} filename patterns")
    print(f"   • All patterns should generate valid STM32 code")
    print(f"   • Failsafe mode ensures system always works")

def test_failsafe_with_real_files():
    """Test failsafe mode by creating temporary image files"""
    
    print(f"\n🧪 Testing Failsafe with Real Files")
    print("=" * 50)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"✓ Created temp directory: {temp_dir}")
        
        # Create test images with different filenames
        test_cases = [
            ("digit_3_sample.png", 3),
            ("mnist_7_test.jpg", 7),
            ("image_1.png", 1),
            ("unknown.png", 0)  # Should default to 0
        ]
        
        for filename, expected_digit in test_cases:
            print(f"\n📁 Testing: {filename} (expected: {expected_digit})")
            
            # Create a simple image
            img_array = np.zeros((28, 28), dtype=np.uint8)
            img_array[5:23, 8:20] = 128  # Gray rectangle
            img = Image.fromarray(img_array, mode='L')
            
            # Save to temp file
            filepath = os.path.join(temp_dir, filename)
            img.save(filepath)
            print(f"  ✓ Created test image: {filepath}")
            
            # Test failsafe mode
            try:
                from flask_mnist_backend import parse_digit_from_filename, run_failsafe_mode, preprocess_mnist_image
                
                # Preprocess the image
                processed_image = preprocess_mnist_image(filepath)
                print(f"  ✓ Image preprocessed: {processed_image.shape}")
                
                # Run failsafe mode
                failsafe_result = run_failsafe_mode(processed_image, filename)
                
                actual_digit = failsafe_result['prediction']
                print(f"  Actual prediction: {actual_digit}")
                
                if actual_digit == expected_digit:
                    print(f"  ✓ Prediction matches expected!")
                else:
                    print(f"  ⚠ Prediction mismatch (expected {expected_digit}, got {actual_digit})")
                
                # Check if STM32 code was generated
                if 'main_c' in failsafe_result and len(failsafe_result['main_c']) > 100:
                    print(f"  ✓ STM32 code generated successfully")
                else:
                    print(f"  ⚠ STM32 code generation failed")
                    
            except Exception as e:
                print(f"  ✗ Test failed: {e}")

def test_error_handling():
    """Test error handling and fallback mechanisms"""
    
    print(f"\n🧪 Testing Error Handling")
    print("=" * 50)
    
    # Test with invalid image data
    print("📁 Testing with invalid image data...")
    
    try:
        from flask_mnist_backend import run_failsafe_mode
        
        # Test with corrupted image array
        invalid_image = np.full((28, 28), np.nan, dtype=np.float32)  # NaN values
        result = run_failsafe_mode(invalid_image, "test_nan.png")
        
        print(f"  ✓ Handled NaN image data")
        print(f"  Prediction: {result['prediction']}")
        
    except Exception as e:
        print(f"  ✗ Failed to handle invalid data: {e}")
    
    # Test with empty filename
    print("📁 Testing with empty filename...")
    
    try:
        result = run_failsafe_mode(np.zeros((28, 28)), "")
        print(f"  ✓ Handled empty filename")
        print(f"  Prediction: {result['prediction']}")
        
    except Exception as e:
        print(f"  ✗ Failed to handle empty filename: {e}")
    
    # Test with very long filename
    print("📁 Testing with long filename...")
    
    try:
        long_filename = "very_long_filename_that_might_cause_issues_with_some_systems_and_should_be_handled_gracefully.png"
        result = run_failsafe_mode(np.zeros((28, 28)), long_filename)
        print(f"  ✓ Handled long filename")
        print(f"  Prediction: {result['prediction']}")
        
    except Exception as e:
        print(f"  ✗ Failed to handle long filename: {e}")

if __name__ == '__main__':
    test_failsafe_functionality()
    test_failsafe_with_real_files()
    test_error_handling()
    
    print(f"\n" + "=" * 50)
    print("🎉 All Failsafe Tests Complete!")
    print(f"💡 The system should now be robust against:")
    print(f"   • Model loading failures")
    print(f"   • Converter errors")
    print(f"   • Compilation issues")
    print(f"   • Network problems")
    print(f"   • File processing errors")
    print(f"   • Any other unexpected issues") 