#!/usr/bin/env python3
"""
Flask backend for MNIST image processing and STM32 neural network inference
"""

import os
import sys
import time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import torch
import torch.nn.functional as F
from werkzeug.utils import secure_filename

# Add the backend src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

try:
    from models import create_model, get_dataset_config
    from converter import convert_model_to_c
except ImportError as e:
    print(f"Warning: Could not import backend modules: {e}")
    print("Make sure you're running from the DeepMetal root directory")

app = Flask(__name__)
CORS(app, origins=['http://localhost:5173'])  # Vite dev server

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model storage
loaded_model = None
model_path = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_mnist_image(image_path):
    """
    Preprocess uploaded image to MNIST format (28x28 grayscale, normalized)
    Returns: numpy array of shape (784,) with values 0-1
    """
    try:
        # Open and convert to grayscale
        img = Image.open(image_path).convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to 0-1 range (MNIST is white digits on black background)
        img_array = img_array / 255.0
        
        # Invert if needed (make sure digits are bright on dark background)
        if np.mean(img_array) > 0.5:
            img_array = 1.0 - img_array
        
        # Flatten to 784 elements
        img_array = img_array.flatten()
        
        return img_array
        
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def load_mnist_model():
    """Load the MNIST model for inference"""
    global loaded_model, model_path
    
    if loaded_model is not None:
        return loaded_model
    
    try:
        # Look for existing models
        models_dir = os.path.join('backend', 'src', 'models')
        model_files = [
            'mnist_hybrid_model.pth',
            'mnist_conv_model.pth', 
            'mnist_linear_model.pth',
            'MNIST_model.pth'
        ]
        
        for model_file in model_files:
            full_path = os.path.join(models_dir, model_file)
            if os.path.exists(full_path):
                print(f"Loading model from: {full_path}")
                loaded_model = torch.load(full_path, map_location='cpu')
                loaded_model.eval()
                model_path = full_path
                return loaded_model
        
        # If no model found, create a new one
        print("No existing model found, creating new MNIST model...")
        from export_model import create_model
        
        model = create_model('mnist', 'hybrid', use_sequential=True)
        model.eval()
        
        # Save the model
        save_path = os.path.join(models_dir, 'mnist_hybrid_model.pth')
        torch.save(model, save_path)
        print(f"Created and saved new model: {save_path}")
        
        loaded_model = model
        model_path = save_path
        return loaded_model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load or create model: {str(e)}")

def run_inference(image_array):
    """
    Run inference on preprocessed image array
    Returns: (prediction, confidence)
    """
    try:
        model = load_mnist_model()
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
        
        # For conv models, reshape to (1, 1, 28, 28)
        if hasattr(model, 'conv1') or 'conv' in str(model).lower():
            input_tensor = input_tensor.view(1, 1, 28, 28)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return prediction, confidence
        
    except Exception as e:
        raise RuntimeError(f"Inference failed: {str(e)}")

def generate_stm32_code(image_array, prediction):
    """Generate STM32 C code with the specific input array"""
    
    # Format the array for C code
    array_elements = []
    for i in range(len(image_array)):
        if i % 8 == 0 and i > 0:
            array_elements.append('\n    ')
        array_elements.append(f'{image_array[i]:.4f}f')
        if i < len(image_array) - 1:
            array_elements.append(', ')
    
    array_code = ''.join(array_elements)
    
    # Generate the complete C code
    c_code = f'''// Generated STM32F446RE Neural Network Code for MNIST Digit Recognition
// Predicted digit: {prediction}
// Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}

#include "../output/model.h"

// Preprocessed MNIST input array (28x28 = 784 elements)
static const float test_digit[784] = {{
    {array_code}
}};

// Forward declarations
void Reset_Handler(void);
void delay_ms(int ms);

// Vector table (proper setup)
__attribute__((section(".isr_vector")))
unsigned int vector_table[] = {{
    0x20020000,                    // Stack pointer
    (unsigned int)Reset_Handler,   // Reset handler
}};

// Delay function
void delay_ms(int ms) {{
    // Approximate delay (assumes 16MHz internal clock)
    for(volatile int i = 0; i < ms * 1000; i++);
}}

// GPIO setup for LED control
void setup_gpio() {{
    // Enable GPIOA clock (bit 0 in AHB1ENR)
    *((volatile unsigned int*)0x40023830) |= 0x00000001;
    
    // Configure PA5 as output (LED pin on STM32F446RE Nucleo)
    volatile unsigned int *moder = (volatile unsigned int*)0x40020000;
    *moder &= ~(0x3 << 10);  // Clear bits 11:10
    *moder |= (0x1 << 10);   // Set as output (01)
}}

// LED control functions
void led_on() {{
    *((volatile unsigned int*)0x40020018) = (1 << 5);  // Set PA5
}}

void led_off() {{
    *((volatile unsigned int*)0x40020018) = (1 << 21); // Reset PA5 (bit 5 + 16)
}}

void led_blink(int count) {{
    for(int i = 0; i < count; i++) {{
        led_on();
        delay_ms(300);
        led_off();
        delay_ms(300);
    }}
}}

// Main program
void main_program() {{
    setup_gpio();
    
    // Startup sequence - 3 fast blinks to indicate system ready
    for(int i = 0; i < 3; i++) {{
        led_on();
        delay_ms(100);
        led_off();
        delay_ms(100);
    }}
    
    delay_ms(1000);  // Pause before starting main loop
    
    // Main inference loop
    while(1) {{
        // Run neural network inference on the preprocessed image
        int prediction = predict(test_digit, 28, 28, 1);
        
        // Ensure prediction is in valid range (0-9)
        if(prediction < 0) prediction = 0;
        if(prediction > 9) prediction = 9;
        
        // Blink LED: (prediction + 1) times to indicate the result
        // For digit 0: 1 blink, digit 1: 2 blinks, ..., digit 9: 10 blinks
        led_blink(prediction + 1);
        
        // Long pause before next inference cycle
        delay_ms(3000);
    }}
}}

// Reset handler - entry point after reset
__attribute__((naked))
void Reset_Handler() {{
    // Minimal startup code - just call main program
    main_program();
    
    // Should never reach here, but safety loop
    while(1);
}}

/*
 * USAGE INSTRUCTIONS:
 * 
 * 1. Make sure you have the neural network model converted to C code in "../output/model.h"
 *    This should contain the predict() function implementation.
 * 
 * 2. Compile this code for STM32F446RE:
 *    arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -O2 -Wall -c main.c -o main.o
 *    arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -O2 -Wall -c model.c -o model.o
 *    arm-none-eabi-ld -T linker.ld main.o model.o -o neural_net.elf
 *    arm-none-eabi-objcopy -O binary neural_net.elf neural_net.bin
 * 
 * 3. Flash to STM32F446RE Nucleo board:
 *    st-flash write neural_net.bin 0x8000000
 * 
 * 4. The LED will blink to indicate the predicted digit:
 *    - 3 fast blinks at startup (system ready)
 *    - Then (prediction + 1) blinks every 3 seconds
 *    - For example: if the model predicts digit 7, LED blinks 8 times
 * 
 * 5. The input image has been preprocessed to 28x28 grayscale and normalized
 *    to match MNIST dataset format.
 */'''
    
    return c_code

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'DeepMetal MNIST Inference API',
        'model_loaded': loaded_model is not None
    })

@app.route('/api/process-mnist', methods=['POST'])
def process_mnist_image():
    """
    Process uploaded MNIST image and return prediction results
    """
    start_time = time.time()
    
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess image
            print(f"Preprocessing image: {filepath}")
            image_array = preprocess_mnist_image(filepath)
            
            # Run inference
            print("Running neural network inference...")
            prediction, confidence = run_inference(image_array)
            
            # Generate STM32 C code
            print("Generating STM32 C code...")
            c_code = generate_stm32_code(image_array, prediction)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Prepare response
            result = {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'processingTime': float(processing_time),
                'imageArray': image_array.tolist(),
                'cCode': c_code,
                'stmOutput': f"LED will blink {prediction + 1} times to indicate digit {prediction}"
            }
            
            print(f"Inference complete: digit={prediction}, confidence={confidence:.2f}, time={processing_time:.1f}ms")
            
            return jsonify({
                'success': True,
                'result': result
            })
        
        finally:
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing image: {error_msg}")
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    try:
        model = load_mnist_model()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get model architecture info
        model_str = str(model)
        
        return jsonify({
            'success': True,
            'info': {
                'modelPath': model_path,
                'totalParameters': total_params,
                'trainableParameters': trainable_params,
                'architecture': model_str[:500] + '...' if len(model_str) > 500 else model_str
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting DeepMetal MNIST Inference API")
    print("📡 Server will be available at: http://localhost:5000")
    print("🧠 Loading neural network model...")
    
    try:
        # Pre-load the model
        load_mnist_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not pre-load model: {e}")
        print("   Model will be loaded on first request")
    
    print("🎯 Ready to process MNIST images!")
    print()
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
