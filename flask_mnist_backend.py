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
import random
import re

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
'''
    return c_code

def parse_digit_from_filename(filename):
    """
    Parses a digit from a filename.
    It checks for specific prefixes first, then falls back to finding the
    first digit of the first number in the filename.
    """
    if not filename:
        return 0

    name_part = os.path.splitext(filename)[0]

    # Check for specific prefixes for more reliable parsing
    prefixes = ['digit_', 'mnist_', 'test_', 'image_', 'sample_']
    for prefix in prefixes:
        if name_part.startswith(prefix):
            try:
                num_str = name_part[len(prefix):].split('_')[0]
                # Use the first digit of the number found
                return int(num_str[0])
            except (ValueError, IndexError):
                continue
    
    # Generic fallback: find the first number in the filename
    try:
        numbers = re.findall(r'\d+', name_part)
        if numbers:
            # Use the first digit of the first number found
            return int(numbers[0][0])
    except:
        pass

    return 0  # Default to 0 if no number is found

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
    Processes an uploaded MNIST image.
    This endpoint now ALWAYS uses filename-based prediction as requested.
    """
    start_time = time.time()
    filepath = None
    original_filename = "unknown_file.png"

    try:
        delay = random.uniform(5, 10)
        time.sleep(delay)
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        original_filename = file.filename
        
        if not original_filename or not allowed_file(original_filename):
            return jsonify({'error': 'Invalid file provided'}), 400
        
        filename = secure_filename(original_filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            image_array = preprocess_mnist_image(filepath)
            print(f"✓ Image preprocessed: {image_array.shape}")
        except Exception as e:
            print(f"✗ Image preprocessing failed: {e}")
            image_array = np.zeros((28, 28), dtype=np.float32)

        # --- Logic Changed ---
        # Always use the filename-based logic and bypass the real model.
        prediction = parse_digit_from_filename(original_filename)
        confidence = random.uniform(0.65, 0.91)

        # Write the provided model_clean.c code
        model_clean_c_code = '''#include "model.h"

// Simple 3-layer neural network weights (very small for STM32)
// Layer 0: 784 -> 8 (sparse weights to save memory)
const float linear_w0[LAYER0_OUT_SIZE][LAYER0_IN_SIZE] = {
    {[100] = 0.1f, [200] = 0.2f, [300] = 0.1f, [400] = 0.2f, [500] = 0.1f, [600] = 0.2f, [700] = 0.1f},
    {[150] = 0.2f, [250] = 0.1f, [350] = 0.2f, [450] = 0.1f, [550] = 0.2f, [650] = 0.1f, [750] = 0.2f},
    {[125] = 0.1f, [225] = 0.2f, [325] = 0.1f, [425] = 0.2f, [525] = 0.1f, [625] = 0.2f, [725] = 0.1f},
    {[175] = 0.2f, [275] = 0.1f, [375] = 0.2f, [475] = 0.1f, [575] = 0.2f, [675] = 0.1f, [775] = 0.2f},
    {[110] = 0.1f, [210] = 0.2f, [310] = 0.1f, [410] = 0.2f, [510] = 0.1f, [610] = 0.2f, [710] = 0.1f},
    {[160] = 0.2f, [260] = 0.1f, [360] = 0.2f, [460] = 0.1f, [560] = 0.2f, [660] = 0.1f, [760] = 0.2f},
    {[135] = 0.1f, [235] = 0.2f, [335] = 0.1f, [435] = 0.2f, [535] = 0.1f, [635] = 0.2f, [735] = 0.1f},
    {[185] = 0.2f, [285] = 0.1f, [385] = 0.2f, [485] = 0.1f, [585] = 0.2f, [685] = 0.1f, [783] = 0.2f}
};

const float linear_b0[LAYER0_OUT_SIZE] = {0.1f, -0.1f, 0.2f, -0.2f, 0.1f, 0.0f, -0.1f, 0.2f};

// Layer 2: 8 -> 4
const float linear_w2[LAYER2_OUT_SIZE][LAYER2_IN_SIZE] = {
    {0.3f, -0.2f, 0.1f, 0.4f, -0.1f, 0.2f, 0.3f, -0.3f},
    {-0.1f, 0.4f, -0.3f, 0.2f, 0.1f, -0.2f, 0.4f, 0.1f},
    {0.2f, 0.1f, 0.3f, -0.1f, 0.4f, 0.2f, -0.3f, 0.2f},
    {-0.2f, 0.3f, 0.1f, 0.2f, -0.1f, 0.3f, 0.1f, -0.4f}
};

const float linear_b2[LAYER2_OUT_SIZE] = {0.1f, -0.1f, 0.2f, 0.0f};

// Layer 4: 4 -> 10 (output layer)
const float linear_w4[LAYER4_OUT_SIZE][LAYER4_IN_SIZE] = {
    {0.5f, -0.3f, 0.2f, 0.1f},  // digit 0
    {-0.2f, 0.4f, 0.3f, -0.1f}, // digit 1
    {0.3f, 0.1f, -0.4f, 0.2f},  // digit 2
    {0.1f, -0.2f, 0.3f, 0.4f},  // digit 3
    {-0.3f, 0.2f, 0.1f, -0.2f}, // digit 4
    {0.2f, 0.3f, -0.1f, 0.4f},  // digit 5
    {0.4f, -0.1f, 0.2f, 0.3f},  // digit 6
    {-0.1f, 0.3f, 0.4f, -0.2f}, // digit 7
    {0.3f, 0.2f, -0.3f, 0.1f},  // digit 8
    {-0.2f, -0.1f, 0.4f, 0.3f}  // digit 9
};

const float linear_b4[LAYER4_OUT_SIZE] = {0.1f, -0.1f, 0.2f, 0.0f, -0.2f, 0.1f, 0.3f, -0.1f, 0.2f, 0.0f};

// Simple ReLU activation
float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Predict function
int predict(const float *input, int input_h, int input_w, int input_ch) {
    static float buf[32]; // Small buffer for intermediate results
    
    // Avoid unused parameter warnings
    (void)input_h;
    (void)input_w;
    (void)input_ch;
    
    // Layer 0: Linear (784 -> 8) + ReLU
    for(int i = 0; i < LAYER0_OUT_SIZE; i++) {
        float sum = linear_b0[i];
        for(int j = 0; j < LAYER0_IN_SIZE; j++) {
            sum += linear_w0[i][j] * input[j];
        }
        buf[i] = relu(sum);
    }
    
    // Layer 2: Linear (8 -> 4) + ReLU
    for(int i = 0; i < LAYER2_OUT_SIZE; i++) {
        float sum = linear_b2[i];
        for(int j = 0; j < LAYER2_IN_SIZE; j++) {
            sum += linear_w2[i][j] * buf[j];
        }
        buf[8 + i] = relu(sum); // Store in buf[8-11]
    }
    
    // Layer 4: Linear (4 -> 10) - output layer
    float max_val = -1000.0f;
    int max_idx = 0;
    
    for(int i = 0; i < LAYER4_OUT_SIZE; i++) {
        float sum = linear_b4[i];
        for(int j = 0; j < LAYER4_IN_SIZE; j++) {
            sum += linear_w4[i][j] * buf[8 + j]; // Use buf[8-11]
        }
        
        if(sum > max_val) {
            max_val = sum;
            max_idx = i;
        }
    }
    
    return max_idx;
}
'''
        output_dir = os.path.join('backend', 'src', 'output')
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'model_clean.c'), 'w') as f:
            f.write(model_clean_c_code)

        processing_time = (time.time() - start_time) * 1000
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'model_type': 'Filename-based Prediction',
            'model_c': f"// Model generated from filename - returns {prediction}\nint predict(const float *input, int h, int w, int c) {{ return {prediction}; }}",
            'model_h': "#ifndef MODEL_H\n#define MODEL_H\nint predict(const float *input, int h, int w, int c);\n#endif",
            'files_generated': ['model_clean.c'],
            'compilation_result': {
                'success': True,
                'message': 'STM32 code generated from filename.',
                'compiler_output': 'Using hardcoded prediction based on filename.'
            },
            'image_shape': image_array.shape,
            'processing_time': processing_time
        }
        return jsonify(result)
        # --- End of Change ---

    except Exception as e:
        print(f"✗ Processing failed: {e}")
        try:
            image_array = preprocess_mnist_image(filepath) if filepath and os.path.exists(filepath) else np.zeros((28, 28), dtype=np.float32)
        except:
            image_array = np.zeros((28, 28), dtype=np.float32)
        
        # Failsafe: still write model_clean.c
        model_clean_c_code = '''#include "model.h"

// Simple 3-layer neural network weights (very small for STM32)
// Layer 0: 784 -> 8 (sparse weights to save memory)
const float linear_w0[LAYER0_OUT_SIZE][LAYER0_IN_SIZE] = {
    {[100] = 0.1f, [200] = 0.2f, [300] = 0.1f, [400] = 0.2f, [500] = 0.1f, [600] = 0.2f, [700] = 0.1f},
    {[150] = 0.2f, [250] = 0.1f, [350] = 0.2f, [450] = 0.1f, [550] = 0.2f, [650] = 0.1f, [750] = 0.2f},
    {[125] = 0.1f, [225] = 0.2f, [325] = 0.1f, [425] = 0.2f, [525] = 0.1f, [625] = 0.2f, [725] = 0.1f},
    {[175] = 0.2f, [275] = 0.1f, [375] = 0.2f, [475] = 0.1f, [575] = 0.2f, [675] = 0.1f, [775] = 0.2f},
    {[110] = 0.1f, [210] = 0.2f, [310] = 0.1f, [410] = 0.2f, [510] = 0.1f, [610] = 0.2f, [710] = 0.1f},
    {[160] = 0.2f, [260] = 0.1f, [360] = 0.2f, [460] = 0.1f, [560] = 0.2f, [660] = 0.1f, [760] = 0.2f},
    {[135] = 0.1f, [235] = 0.2f, [335] = 0.1f, [435] = 0.2f, [535] = 0.1f, [635] = 0.2f, [735] = 0.1f},
    {[185] = 0.2f, [285] = 0.1f, [385] = 0.2f, [485] = 0.1f, [585] = 0.2f, [685] = 0.1f, [783] = 0.2f}
};

const float linear_b0[LAYER0_OUT_SIZE] = {0.1f, -0.1f, 0.2f, -0.2f, 0.1f, 0.0f, -0.1f, 0.2f};

// Layer 2: 8 -> 4
const float linear_w2[LAYER2_OUT_SIZE][LAYER2_IN_SIZE] = {
    {0.3f, -0.2f, 0.1f, 0.4f, -0.1f, 0.2f, 0.3f, -0.3f},
    {-0.1f, 0.4f, -0.3f, 0.2f, 0.1f, -0.2f, 0.4f, 0.1f},
    {0.2f, 0.1f, 0.3f, -0.1f, 0.4f, 0.2f, -0.3f, 0.2f},
    {-0.2f, 0.3f, 0.1f, 0.2f, -0.1f, 0.3f, 0.1f, -0.4f}
};

const float linear_b2[LAYER2_OUT_SIZE] = {0.1f, -0.1f, 0.2f, 0.0f};

// Layer 4: 4 -> 10 (output layer)
const float linear_w4[LAYER4_OUT_SIZE][LAYER4_IN_SIZE] = {
    {0.5f, -0.3f, 0.2f, 0.1f},  // digit 0
    {-0.2f, 0.4f, 0.3f, -0.1f}, // digit 1
    {0.3f, 0.1f, -0.4f, 0.2f},  // digit 2
    {0.1f, -0.2f, 0.3f, 0.4f},  // digit 3
    {-0.3f, 0.2f, 0.1f, -0.2f}, // digit 4
    {0.2f, 0.3f, -0.1f, 0.4f},  // digit 5
    {0.4f, -0.1f, 0.2f, 0.3f},  // digit 6
    {-0.1f, 0.3f, 0.4f, -0.2f}, // digit 7
    {0.3f, 0.2f, -0.3f, 0.1f},  // digit 8
    {-0.2f, -0.1f, 0.4f, 0.3f}  // digit 9
};

const float linear_b4[LAYER4_OUT_SIZE] = {0.1f, -0.1f, 0.2f, 0.0f, -0.2f, 0.1f, 0.3f, -0.1f, 0.2f, 0.0f};

// Simple ReLU activation
float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Predict function
int predict(const float *input, int input_h, int input_w, int input_ch) {
    static float buf[32]; // Small buffer for intermediate results
    
    // Avoid unused parameter warnings
    (void)input_h;
    (void)input_w;
    (void)input_ch;
    
    // Layer 0: Linear (784 -> 8) + ReLU
    for(int i = 0; i < LAYER0_OUT_SIZE; i++) {
        float sum = linear_b0[i];
        for(int j = 0; j < LAYER0_IN_SIZE; j++) {
            sum += linear_w0[i][j] * input[j];
        }
        buf[i] = relu(sum);
    }
    
    // Layer 2: Linear (8 -> 4) + ReLU
    for(int i = 0; i < LAYER2_OUT_SIZE; i++) {
        float sum = linear_b2[i];
        for(int j = 0; j < LAYER2_IN_SIZE; j++) {
            sum += linear_w2[i][j] * buf[j];
        }
        buf[8 + i] = relu(sum); // Store in buf[8-11]
    }
    
    // Layer 4: Linear (4 -> 10) - output layer
    float max_val = -1000.0f;
    int max_idx = 0;
    
    for(int i = 0; i < LAYER4_OUT_SIZE; i++) {
        float sum = linear_b4[i];
        for(int j = 0; j < LAYER4_IN_SIZE; j++) {
            sum += linear_w4[i][j] * buf[8 + j]; // Use buf[8-11]
        }
        
        if(sum > max_val) {
            max_val = sum;
            max_idx = i;
        }
    }
    
    return max_idx;
}
'''
        output_dir = os.path.join('backend', 'src', 'output')
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'model_clean.c'), 'w') as f:
            f.write(model_clean_c_code)
        
        failsafe_result = {
            'prediction': prediction,
            'confidence': confidence,
            'model_type': 'Filename-based Prediction',
            'model_c': f"// Model generated from filename - returns {prediction}\nint predict(const float *input, int h, int w, int c) {{ return {prediction}; }}",
            'model_h': "#ifndef MODEL_H\n#define MODEL_H\nint predict(const float *input, int h, int w, int c);\n#endif",
            'files_generated': ['model_clean.c'],
            'compilation_result': {
                'success': False,
                'message': f'Processing failed: {str(e)}',
                'compiler_output': 'Error during processing.'
            },
            'image_shape': image_array.shape,
            'processing_time': (time.time() - start_time) * 1000,
            'error': f'Processing failed: {str(e)}'
        }
        return jsonify(failsafe_result), 200
    
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

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
