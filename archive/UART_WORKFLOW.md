# UART-Based STM32 MNIST Workflow

This document explains the complete workflow for sending image data to STM32 via UART and receiving prediction results.

## Overview

The workflow enables bidirectional communication between a PC and STM32 microcontroller:
1. **PC → STM32**: Send 784 float values (28x28 MNIST image)
2. **STM32**: Run neural network inference on the image
3. **STM32 → PC**: Send back prediction digit and confidence

## Hardware Setup

### STM32 Connections
- **PA9 (UART1_TX)**: Connect to USB-to-UART converter RX
- **PA10 (UART1_RX)**: Connect to USB-to-UART converter TX
- **GND**: Connect to USB-to-UART converter GND

### UART Settings
- **Baud Rate**: 115200
- **Data Bits**: 8
- **Parity**: None
- **Stop Bits**: 1
- **Flow Control**: None

## Software Components

### 1. STM32 Code (`main.c`)
The generated `main.c` includes:
- UART initialization and communication functions
- Image data reception from serial monitor
- Neural network inference using the converted model
- Result transmission back over UART

### 2. Python Communication Script (`uart_communication.py`)
Handles PC-side communication:
- Connects to STM32 via serial port
- Sends image data as comma-separated values
- Receives and parses prediction results

## Complete Workflow

### Step 1: Generate STM32 Code
```bash
# The Flask backend generates:
# - model.c (neural network implementation)
# - model.h (header file)
# - main.c (UART communication + inference)
```

### Step 2: Compile and Flash
```bash
# Compile the code
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O2 -c model.c -o model.o
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O2 -c main.c -o main.o

# Link and create binary
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -T linker.ld model.o main.o -o neural_network.elf
arm-none-eabi-objcopy -O binary neural_network.elf neural_network.bin

# Flash to STM32
st-flash write neural_network.bin 0x08000000
```

### Step 3: Run UART Communication
```bash
# Install requirements
pip install -r uart_requirements.txt

# Run communication script
python uart_communication.py COM3  # Replace COM3 with your port

# Or send a specific image
python uart_communication.py COM3 path/to/image.png
```

## Data Format

### Image Data (PC → STM32)
```
-0.424000,0.123456,0.789012,...,0.234567
```
- 784 comma-separated float values
- Each value represents one pixel (28x28 = 784 pixels)
- Values are normalized MNIST format

### Results (STM32 → PC)
```
=== RESULTS ===
Predicted digit: 3
Confidence: 85.67%
================
```

## Example Usage

### 1. Using the Flask Backend
```python
# Upload image through web interface
# Backend generates STM32 code with UART communication
# Download and flash the generated code
```

### 2. Manual Testing
```python
from uart_communication import STM32UARTCommunicator
import numpy as np

# Create communicator
comm = STM32UARTCommunicator(port='COM3')

# Connect
if comm.connect():
    # Create test image
    test_image = np.random.rand(28, 28).astype(np.float32)
    
    # Send to STM32
    comm.send_image_data(test_image)
    
    # Read results
    results = comm.read_results()
    if results:
        print(f"Prediction: {results['prediction']}")
        print(f"Confidence: {results['confidence']}%")
    
    comm.disconnect()
```

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check COM port number
   - Verify USB-to-UART driver installation
   - Ensure STM32 is powered and connected

2. **No Response from STM32**
   - Check baud rate (115200)
   - Verify UART pins (PA9/PA10)
   - Check if STM32 code is flashed correctly

3. **Garbled Data**
   - Check baud rate mismatch
   - Verify voltage levels (3.3V for STM32)
   - Check wiring connections

### Debug Commands

```bash
# List available COM ports
python -c "import serial.tools.list_ports; [print(f'{p.device}: {p.description}') for p in serial.tools.list_ports.comports()]"

# Test serial communication
python uart_communication.py COM3
```

## Advanced Features

### 1. Real-time Image Processing
The STM32 can continuously receive new images and process them in real-time.

### 2. Batch Processing
Send multiple images sequentially and collect all results.

### 3. Confidence Threshold
Add confidence thresholds to filter low-confidence predictions.

### 4. Model Switching
Implement multiple models and switch between them via UART commands.

## Performance Considerations

- **Inference Time**: ~10-100ms depending on model size
- **UART Transfer Time**: ~1-2 seconds for 784 float values
- **Memory Usage**: ~4KB for image buffer + model weights
- **Power Consumption**: Optimized for battery operation

## Future Enhancements

1. **Binary Protocol**: Use binary data instead of ASCII for faster transfer
2. **Compression**: Compress image data before transmission
3. **DMA**: Use DMA for faster UART transfers
4. **Interrupts**: Use UART interrupts for better performance
5. **Multiple UARTs**: Use multiple UART channels for parallel processing 