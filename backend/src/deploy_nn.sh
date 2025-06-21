#!/bin/bash
# deploy_neural_network.sh - deploy working neural network after LED test passes

echo "🧠 STM32F446RE Neural Network Deployment"
echo "========================================"

cd stm32_project

echo "📋 Creating robust neural network firmware..."

# Create working main program
cat > neural_main.c << 'EOF'
// Working STM32F446RE Neural Network
#include "../output/model.h"

// Test MNIST pattern
static const float test_digit[784] = {
    [0 ... 783] = 0.0f,
    // Add a simple digit pattern (could be improved with real MNIST data)
    [100] = 0.8f, [101] = 0.9f, [102] = 0.8f,
    [128] = 0.7f, [129] = 0.9f, [130] = 0.7f,
};

// Forward declarations
void Reset_Handler(void);
void delay_ms(int ms);

// Vector table (proper setup)
__attribute__((section(".isr_vector")))
unsigned int vector_table[] = {
    0x20020000,                    // Stack pointer
    (unsigned int)Reset_Handler,   // Reset handler
};

// Delay function
void delay_ms(int ms) {
    // Approximate delay (assumes 16MHz internal clock)
    for(volatile int i = 0; i < ms * 1000; i++);
}

// GPIO setup
void setup_gpio() {
    // Enable GPIOA clock
    *((volatile unsigned int*)0x40023830) |= 0x00000001;
    
    // Configure PA5 as output
    volatile unsigned int *moder = (volatile unsigned int*)0x40020000;
    *moder &= ~(0x3 << 10);  // Clear bits
    *moder |= (0x1 << 10);   // Set as output
}

// LED control
void led_on() {
    *((volatile unsigned int*)0x40020018) = (1 << 5);
}

void led_off() {
    *((volatile unsigned int*)0x40020018) = (1 << 21);
}

void led_blink(int count) {
    for(int i = 0; i < count; i++) {
        led_on();
        delay_ms(300);
        led_off();
        delay_ms(300);
    }
}

// Main program
void main_program() {
    setup_gpio();
    
    // Startup sequence - 3 fast blinks
    for(int i = 0; i < 3; i++) {
        led_on();
        delay_ms(100);
        led_off();
        delay_ms(100);
    }
    
    delay_ms(1000);  // Pause
    
    // Main loop
    while(1) {
        // Run neural network inference
        int prediction = predict(test_digit, 28, 28, 1);
        
        // Ensure valid range
        if(prediction < 0) prediction = 0;
        if(prediction > 9) prediction = 9;
        
        // Blink LED: prediction + 1 times
        led_blink(prediction + 1);
        
        // Long pause before next prediction
        delay_ms(3000);
    }
}

// Reset handler
__attribute__((naked))
void Reset_Handler() {
    // Basic startup (minimal for this demo)
    main_program();
    
    // Should never reach here, but just in case
    while(1);
}
EOF

# Create working linker script
cat > neural.ld << 'EOF'
MEMORY
{
    FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 512K
    RAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 128K
}

SECTIONS
{
    .isr_vector : {
        . = ALIGN(4);
        *(.isr_vector)
        . = ALIGN(4);
    } > FLASH
    
    .text : {
        . = ALIGN(4);
        *(.text*)
        . = ALIGN(4);
    } > FLASH
    
    .rodata : {
        . = ALIGN(4);
        *(.rodata*)
        . = ALIGN(4);
    } > FLASH
    
    /* Data and BSS sections for neural network weights */
    _sidata = LOADADDR(.data);
    
    .data : {
        . = ALIGN(4);
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT> FLASH
    
    .bss : {
        . = ALIGN(4);
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM
}
EOF

echo "🔨 Compiling neural network firmware..."
arm-none-eabi-gcc \
    -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard \
    -Os -Wall \
    -ffunction-sections -fdata-sections \
    -T neural.ld \
    -Wl,--gc-sections \
    --specs=nano.specs --specs=nosys.specs \
    neural_main.c ../output/model.c \
    -o neural_net.elf

if [ $? -eq 0 ]; then
    echo "✅ Neural network compilation successful"
    
    # Create binary
    arm-none-eabi-objcopy -O binary neural_net.elf neural_net.bin
    
    # Show size
    echo "📊 Firmware size:"
    ls -lh neural_net.bin
    arm-none-eabi-size neural_net.elf 2>/dev/null || echo "   (size analysis unavailable)"
    
    echo ""
    echo "📱 Flashing neural network..."
    st-flash write neural_net.bin 0x08000000
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 NEURAL NETWORK DEPLOYED!"
        echo "=========================="
        echo "👀 Expected behavior:"
        echo "   1. 3 quick blinks on startup"
        echo "   2. Neural network runs"
        echo "   3. LED blinks N+1 times (N = prediction)"
        echo "   4. 3-second pause"
        echo "   5. Repeat from step 2"
        echo ""
        echo "🧠 Your neural network is running on STM32!"
    else
        echo "❌ Flash failed"
    fi
else
    echo "❌ Neural network compilation failed"
fi