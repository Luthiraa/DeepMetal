#!/bin/bash
# simple_led_test.sh - minimal LED test for STM32F446RE debugging

echo "🔍 STM32F446RE LED Debug Test"
echo "============================="

cd stm32_project

echo "📋 Creating ultra-minimal LED test..."

# Create the simplest possible working LED blink
cat > test_led.c << 'EOF'
// Ultra-minimal STM32F446RE LED test
// This MUST work if hardware is OK

// Vector table
__attribute__((section(".isr_vector")))
unsigned int vector_table[] = {
    0x20020000,   // Stack pointer (128KB RAM end)
    0x08000009,   // Reset handler (thumb mode, +1)
};

// Simple blink function  
void blink_forever() {
    // Enable GPIOA clock (RCC_AHB1ENR register)
    *((volatile unsigned int*)0x40023830) |= 0x00000001;
    
    // Configure PA5 as output (GPIOA_MODER register)
    volatile unsigned int *moder = (volatile unsigned int*)0x40020000;
    *moder &= ~(0x3 << 10);  // Clear bits 10-11
    *moder |= (0x1 << 10);   // Set bit 10 (output mode)
    
    // LED control register
    volatile unsigned int *bsrr = (volatile unsigned int*)0x40020018;
    
    // Infinite blink loop
    while(1) {
        // LED ON
        *bsrr = (1 << 5);
        
        // Delay
        for(volatile int i = 0; i < 1000000; i++);
        
        // LED OFF  
        *bsrr = (1 << 21);
        
        // Delay
        for(volatile int i = 0; i < 1000000; i++);
    }
}

// Reset handler - called at startup
__attribute__((naked))
void reset_handler() {
    // Jump to blink function
    blink_forever();
}
EOF

# Create minimal linker script
cat > minimal.ld << 'EOF'
MEMORY
{
    FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 512K
    RAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 128K
}

SECTIONS
{
    .isr_vector : { *(.isr_vector) } > FLASH
    .text : { *(.text*) } > FLASH
}
EOF

echo "🔨 Compiling minimal test..."
arm-none-eabi-gcc \
    -mcpu=cortex-m4 -mthumb \
    -Os -Wall \
    -nostdlib -nostartfiles \
    -T minimal.ld \
    test_led.c -o test_led.elf

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful"
    
    # Create binary
    arm-none-eabi-objcopy -O binary test_led.elf test_led.bin
    
    # Show size
    ls -l test_led.bin
    
    echo "📱 Flashing minimal test..."
    st-flash write test_led.bin 0x08000000
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎯 MINIMAL TEST FLASHED!"
        echo "========================"
        echo "👀 Look at your STM32 board NOW:"
        echo "   - Green LED should blink every 2 seconds"
        echo "   - If NO blinking = hardware/connection issue"
        echo "   - If blinking = STM32 works, software issue"
        echo ""
        echo "🔍 Debugging guide:"
        echo "   NO LED = Check power, connections, board type"
        echo "   LED BLINKS = Hardware OK, proceed with neural network"
    else
        echo "❌ Flash failed"
    fi
else
    echo "❌ Compilation failed"
fi