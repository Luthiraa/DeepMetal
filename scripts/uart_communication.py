#!/usr/bin/env python3
"""
UART Communication Script for STM32 MNIST Neural Network
Sends image data to STM32 and receives prediction results
"""

import serial
import time
import numpy as np
from PIL import Image
import sys
import os

class STM32UARTCommunicator:
    def __init__(self, port='COM3', baudrate=115200, timeout=5):
        """Initialize UART communication with STM32"""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        
    def connect(self):
        """Connect to STM32 via UART"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            print(f"✓ Connected to STM32 on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"✗ Failed to connect to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from STM32"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("✓ Disconnected from STM32")
    
    def read_until_prompt(self, prompt="Send 784"):
        """Read from UART until we see the prompt"""
        buffer = ""
        while True:
            if self.ser.in_waiting > 0:
                char = self.ser.read().decode('ascii', errors='ignore')
                buffer += char
                if prompt in buffer:
                    return buffer
            time.sleep(0.01)
    
    def send_image_data(self, image_array):
        """Send image data to STM32 as comma-separated values"""
        if not self.ser or not self.ser.is_open:
            print("✗ Not connected to STM32")
            return False
        
        # Flatten and convert to string
        flat_image = image_array.flatten()
        image_string = ",".join([f"{x:.6f}" for x in flat_image])
        
        print(f"Sending {len(flat_image)} values to STM32...")
        
        # Send the data
        self.ser.write(image_string.encode('ascii'))
        self.ser.write(b'\r\n')  # End with newline
        self.ser.flush()
        
        print("✓ Image data sent")
        return True
    
    def read_results(self):
        """Read prediction results from STM32"""
        if not self.ser or not self.ser.is_open:
            print("✗ Not connected to STM32")
            return None
        
        print("Reading results from STM32...")
        
        # Wait for results
        time.sleep(2)  # Give STM32 time to process
        
        results = {}
        buffer = ""
        
        # Read for a few seconds to get the results
        start_time = time.time()
        while time.time() - start_time < 10:  # 10 second timeout
            if self.ser.in_waiting > 0:
                char = self.ser.read().decode('ascii', errors='ignore')
                buffer += char
                
                # Look for prediction line
                if "Predicted digit:" in buffer and "Confidence:" in buffer:
                    lines = buffer.split('\r\n')
                    for line in lines:
                        if "Predicted digit:" in line:
                            try:
                                digit = int(line.split(":")[1].strip())
                                results['prediction'] = digit
                            except:
                                pass
                        elif "Confidence:" in line:
                            try:
                                confidence_str = line.split(":")[1].strip().replace("%", "")
                                confidence = float(confidence_str)
                                results['confidence'] = confidence
                            except:
                                pass
                    
                    if 'prediction' in results and 'confidence' in results:
                        return results
            
            time.sleep(0.01)
        
        print("✗ Timeout waiting for results")
        return None
    
    def send_image_file(self, image_path):
        """Send image from file to STM32"""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('L').resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Normalize like MNIST
            img_array = (img_array - 0.1307) / 0.3081
            
            # Invert if background is white
            if np.mean(img_array) > 0:
                img_array = -img_array
            
            print(f"✓ Image loaded: {image_path}")
            print(f"  Shape: {img_array.shape}")
            print(f"  Range: [{img_array.min():.3f}, {img_array.max():.3f}]")
            
            # Send to STM32
            return self.send_image_data(img_array)
            
        except Exception as e:
            print(f"✗ Failed to load image {image_path}: {e}")
            return False

def main():
    """Main function to demonstrate UART communication"""
    print("=== STM32 UART Communication Demo ===")
    
    # Configuration
    port = 'COM3'  # Change this to your STM32's COM port
    image_path = None
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        port = sys.argv[1]
    if len(sys.argv) > 2:
        image_path = sys.argv[2]
    
    # Create communicator
    comm = STM32UARTCommunicator(port=port)
    
    try:
        # Connect to STM32
        if not comm.connect():
            print("Available ports:")
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                print(f"  {p.device}: {p.description}")
            return
        
        # Wait for STM32 to be ready
        print("Waiting for STM32 to be ready...")
        comm.read_until_prompt()
        
        if image_path and os.path.exists(image_path):
            # Send image from file
            print(f"\nSending image: {image_path}")
            if comm.send_image_file(image_path):
                results = comm.read_results()
                if results:
                    print(f"\n=== STM32 Results ===")
                    print(f"Prediction: {results['prediction']}")
                    print(f"Confidence: {results['confidence']:.2f}%")
                else:
                    print("✗ Failed to read results")
        else:
            # Send test pattern
            print("\nSending test pattern...")
            test_image = np.zeros((28, 28), dtype=np.float32)
            test_image[5:23, 8:20] = 0.8  # Simple rectangle pattern
            test_image[10:18, 10:18] = 0.2  # Inner area
            test_image = (test_image - 0.1307) / 0.3081  # Normalize
            
            if comm.send_image_data(test_image):
                results = comm.read_results()
                if results:
                    print(f"\n=== STM32 Results ===")
                    print(f"Prediction: {results['prediction']}")
                    print(f"Confidence: {results['confidence']:.2f}%")
                else:
                    print("✗ Failed to read results")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        comm.disconnect()

if __name__ == '__main__':
    main() 