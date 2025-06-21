#!/bin/bash
# setup_stm32.sh - sets up development environment for stm32f446re

echo "🚀 setting up stm32f446re development environment"
echo "=================================================="

# detect os
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "❌ unsupported os: $OSTYPE"
    exit 1
fi

echo "🔍 detected os: $OS"

# install arm toolchain
echo "📦 installing arm-none-eabi toolchain..."

if [ "$OS" == "linux" ]; then
    # ubuntu/debian
    if command -v apt >/dev/null 2>&1; then
        sudo apt update
        sudo apt install -y gcc-arm-none-eabi gdb-arm-none-eabi
        echo "✅ installed arm toolchain via apt"
    
    # arch linux
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S arm-none-eabi-gcc arm-none-eabi-gdb
        echo "✅ installed arm toolchain via pacman"
    
    # fedora/rhel
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install arm-none-eabi-gcc-cs arm-none-eabi-gdb
        echo "✅ installed arm toolchain via dnf"
    else
        echo "❌ package manager not detected"
        echo "📝 install manually: https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm"
    fi

elif [ "$OS" == "macos" ]; then
    # macos with homebrew
    if command -v brew >/dev/null 2>&1; then
        brew install --cask gcc-arm-embedded
        echo "✅ installed arm toolchain via homebrew"
    else
        echo "❌ homebrew not found"
        echo "📝 install homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "📝 then run: brew install --cask gcc-arm-embedded"
    fi
fi

# verify installation
echo "🧪 verifying arm toolchain installation..."

if command -v arm-none-eabi-gcc >/dev/null 2>&1; then
    echo "✅ arm-none-eabi-gcc found"
    arm-none-eabi-gcc --version | head -1
else
    echo "❌ arm-none-eabi-gcc not found"
    echo "💡 add to path or install manually"
fi

if command -v arm-none-eabi-g++ >/dev/null 2>&1; then
    echo "✅ arm-none-eabi-g++ found"
else
    echo "❌ arm-none-eabi-g++ not found"
fi

# install stm32 specific tools
echo "🔧 installing stm32 development tools..."

if [ "$OS" == "linux" ]; then
    # stlink tools for flashing/debugging
    if command -v apt >/dev/null 2>&1; then
        sudo apt install -y stlink-tools
        echo "✅ installed stlink-tools"
    fi
    
    # openocd for debugging
    if command -v apt >/dev/null 2>&1; then
        sudo apt install -y openocd
        echo "✅ installed openocd"
    fi

elif [ "$OS" == "macos" ]; then
    if command -v brew >/dev/null 2>&1; then
        brew install stlink openocd
        echo "✅ installed stlink and openocd"
    fi
fi

# create stm32f446re specific makefile template
echo "📝 creating stm32f446re makefile template..."

cat > Makefile.stm32f446re << 'EOF'
# Makefile for STM32F446RE with neural network inference

# Target settings
TARGET = stm32f446re_nn_inference
MCU = cortex-m4

# Toolchain
CC = arm-none-eabi-gcc
CXX = arm-none-eabi-g++
OBJCOPY = arm-none-eabi-objcopy
SIZE = arm-none-eabi-size

# STM32F446RE specific settings
CPU = -mcpu=$(MCU)
FPU = -mfpu=fpv4-sp-d16
FLOAT-ABI = -mfloat-abi=hard

# Compiler flags
CFLAGS = $(CPU) -mthumb $(FPU) $(FLOAT-ABI)
CFLAGS += -DSTM32F446xx -DUSE_HAL_DRIVER
CFLAGS += -Wall -Wextra -O3 -g3
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -MMD -MP

# Linker flags
LDFLAGS = $(CPU) -mthumb $(FPU) $(FLOAT-ABI)
LDFLAGS += -specs=nano.specs -specs=nosys.specs
LDFLAGS += -Wl,--gc-sections -static -Wl,--start-group -lc -lm -Wl,--end-group

# Source files
SRCS = main.c output/model.c
OBJS = $(SRCS:.c=.o)

# Build rules
all: $(TARGET).elf $(TARGET).hex $(TARGET).bin
	$(SIZE) $(TARGET).elf

$(TARGET).elf: $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $@

$(TARGET).hex: $(TARGET).elf
	$(OBJCOPY) -O ihex $< $@

$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary -S $< $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJS) $(TARGET).elf $(TARGET).hex $(TARGET).bin

flash: $(TARGET).bin
	st-flash write $(TARGET).bin 0x8000000

.PHONY: all clean flash
EOF

echo "✅ created Makefile.stm32f446re"

# create example main.c for stm32f446re
echo "📝 creating example main.c for stm32f446re..."

cat > main_stm32f446re.c << 'EOF'
/* main_stm32f446re.c - example neural network inference on stm32f446re */

#include "output/model.h"
#include <stdint.h>

// STM32F446RE specific includes (add your HAL includes here)
// #include "stm32f4xx_hal.h"

// Example sensor data (replace with actual sensor readings)
float sensor_data[784] = {0.0f}; // MNIST input size

// System initialization (implement based on your HAL)
void SystemClock_Config(void);
void GPIO_Init(void);
void UART_Init(void);

// Simple UART print function (implement based on your HAL)
void uart_print(const char* str) {
    // TODO: implement UART transmission
    (void)str; // suppress unused warning
}

int main(void) {
    // Initialize STM32F446RE peripherals
    // HAL_Init();
    // SystemClock_Config();
    // GPIO_Init();
    // UART_Init();
    
    uart_print("STM32F446RE Neural Network Inference Starting...\r\n");
    
    while (1) {
        // Read sensor data (replace with actual sensor reading code)
        // read_sensors(sensor_data);
        
        // Run neural network inference
        int prediction = predict(sensor_data, 28, 28, 1);
        
        // Handle prediction result
        switch (prediction) {
            case 0: uart_print("Predicted: 0\r\n"); break;
            case 1: uart_print("Predicted: 1\r\n"); break;
            case 2: uart_print("Predicted: 2\r\n"); break;
            case 3: uart_print("Predicted: 3\r\n"); break;
            case 4: uart_print("Predicted: 4\r\n"); break;
            case 5: uart_print("Predicted: 5\r\n"); break;
            case 6: uart_print("Predicted: 6\r\n"); break;
            case 7: uart_print("Predicted: 7\r\n"); break;
            case 8: uart_print("Predicted: 8\r\n"); break;
            case 9: uart_print("Predicted: 9\r\n"); break;
            default: uart_print("Prediction error\r\n"); break;
        }
        
        // Wait before next inference
        // HAL_Delay(1000); // 1 second delay
    }
}

// Implement these functions based on your STM32F446RE HAL setup
void SystemClock_Config(void) {
    // Configure system clock to 180MHz
}

void GPIO_Init(void) {
    // Initialize GPIO pins
}

void UART_Init(void) {
    // Initialize UART for debugging
}
EOF

echo "✅ created main_stm32f446re.c example"

# create memory usage analysis script
echo "📝 creating memory analysis script..."

cat > analyze_memory.py << 'EOF'
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
EOF

chmod +x analyze_memory.py
echo "✅ created analyze_memory.py"

echo
echo "🎉 stm32f446re development environment setup complete!"
echo
echo "📝 next steps:"
echo "1. generate neural network: python converter.py models/your_model.pth"
echo "2. analyze memory usage: python analyze_memory.py"
echo "3. build for stm32f446re: make -f Makefile.stm32f446re"
echo "4. flash to device: make -f Makefile.stm32f446re flash"
echo
echo "🔧 files created:"
echo "  - Makefile.stm32f446re (build system)"
echo "  - main_stm32f446re.c (example application)"
echo "  - analyze_memory.py (memory analysis tool)"
echo
echo "💡 remember to:"
echo "  - add your STM32 HAL drivers"
echo "  - configure system clock and peripherals"
echo "  - implement sensor data acquisition"
echo "  - adjust linker script for your memory layout"