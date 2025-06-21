#!/bin/bash
# deploy_to_stm32.sh - simple STM32F446RE deployment using existing files

echo "🚀 Deploying Neural Network to STM32F446RE"
echo "=========================================="

# Check if model files exist
if [ ! -f "output/model.h" ] || [ ! -f "output/model.c" ]; then
    echo "❌ Model files missing!"
    echo "🔧 Creating nano model..."
    python export_model.py --model-type linear --model-size nano --epochs 1
    python converter.py models/mnist_linear_model.pth
fi

# Check STM32 tools
if ! command -v arm-none-eabi-gcc &> /dev/null; then
    echo "❌ ARM GCC not found! Install with:"
    echo "   sudo apt install gcc-arm-none-eabi"
    exit 1
fi

if ! command -v st-flash &> /dev/null; then
    echo "❌ ST-Flash not found! Install with:"
    echo "   sudo apt install stlink-tools"
    exit 1
fi

# Setup STM32 project
echo "📁 Setting up STM32 project..."
cd stm32_project

# Copy the simple files
echo "📋 Creating STM32 files..."

# Create main.c
cat > main.c << 'EOF'
#include "../output/model.h"

static const float test_digit[784] = {
    [0 ... 783] = 0.0f,  // Initialize all to 0
    [100] = 0.8f, [101] = 0.9f, [102] = 0.8f  // Add simple pattern
};

int main(void) {
    // Enable GPIOA clock
    *((volatile unsigned int*)0x40023830) |= 0x1;
    // Set PA5 as output  
    *((volatile unsigned int*)0x40020000) |= (1 << 10);
    
    volatile unsigned int *LED = (volatile unsigned int*)0x40020018;
    
    while (1) {
        int prediction = predict(test_digit, 28, 28, 1);
        
        // Blink LED: prediction + 1 times
        for (int i = 0; i <= prediction; i++) {
            *LED = (1 << 5);   // LED on
            for (volatile int d = 0; d < 200000; d++);
            *LED = (1 << 21);  // LED off
            for (volatile int d = 0; d < 200000; d++);
        }
        
        // Long pause
        for (volatile int d = 0; d < 2000000; d++);
    }
}

// Vector table (minimal)
__attribute__((section(".isr_vector")))
void (* const vector_table[])(void) = {
    (void (*)(void))((unsigned long)0x20020000),  // Stack pointer
    main,  // Reset handler
};
EOF

# Create simple Makefile
cat > Makefile << 'EOF'
PROJECT = neural_net
BUILD_DIR = build

PREFIX = arm-none-eabi-
CC = $(PREFIX)gcc
OBJCOPY = $(PREFIX)objcopy
SIZE = $(PREFIX)size

MCU = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
CFLAGS = $(MCU) -DSTM32F446xx -Os -Wall -ffunction-sections -fdata-sections
LDFLAGS = $(MCU) -T linker.ld -Wl,--gc-sections --specs=nano.specs --specs=nosys.specs

SOURCES = main.c ../output/model.c

$(BUILD_DIR):
	mkdir -p $@

all: $(BUILD_DIR)/$(PROJECT).bin size

$(BUILD_DIR)/$(PROJECT).elf: $(SOURCES) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/$(PROJECT).bin: $(BUILD_DIR)/$(PROJECT).elf
	$(OBJCOPY) -O binary $< $@

size: $(BUILD_DIR)/$(PROJECT).elf
	@echo "📊 Memory Usage:"
	@$(SIZE) $< 2>/dev/null | tail -1 | awk '{printf "Flash: %d bytes\nRAM:   %d bytes\n", $$1+$$2, $$2+$$3}' || echo "Size unavailable"

flash: $(BUILD_DIR)/$(PROJECT).bin
	@echo "📱 Flashing to STM32..."
	st-flash write $< 0x08000000

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all flash size clean
EOF

# Create simple linker script
cat > linker.ld << 'EOF'
MEMORY
{
    FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 512K
    RAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 128K
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
    .isr_vector : { KEEP(*(.isr_vector)) } >FLASH
    .text : { *(.text*) } >FLASH
    .rodata : { *(.rodata*) } >FLASH
    
    _sidata = LOADADDR(.data);
    .data : { 
        _sdata = .;
        *(.data*)
        _edata = .;
    } >RAM AT> FLASH
    
    .bss : {
        _sbss = .;
        *(.bss*)
        _ebss = .;
    } >RAM
}
EOF

echo "✅ STM32 project setup complete"

# Build
echo "🔨 Building..."
if make all; then
    echo "✅ Build successful!"
    
    # Check if STM32 is connected
    if lsusb | grep -i "st.*link\|stm" &> /dev/null; then
        echo "✅ STM32 detected"
        
        # Flash
        echo "📱 Flashing..."
        if make flash; then
            echo "🎉 SUCCESS! Neural network deployed to STM32!"
            echo ""
            echo "👀 Watch your STM32 board:"
            echo "   - Green LED (PA5) will blink"
            echo "   - Number of blinks = neural network prediction + 1"
            echo "   - Pattern repeats every few seconds"
            echo ""
            echo "🧠 Your neural network is now running on hardware!"
        else
            echo "❌ Flash failed - check connections"
        fi
    else
        echo "⚠️ STM32 not detected - connect via USB and try 'make flash'"
    fi
else
    echo "❌ Build failed"
fi