#!/bin/bash
# simple_blink_test.sh - Test basic STM32 functionality
echo "🔍 Testing Basic STM32 Functionality"
echo "===================================="

cd stm32_project

# Create ultra-simple blink program with fixed vector table
cat > simple_blink.c << 'ENDFILE'
void Reset_Handler(void);

__attribute__((section(".isr_vector")))
void (* const vector_table[])(void) = {
    (void (*)(void))0x20020000,
    Reset_Handler,
};

void delay(void) {
    for(volatile int i = 0; i < 1000000; i++);
}

void Reset_Handler(void) {
    // Enable GPIOA clock
    *((volatile unsigned int*)0x40023830) |= (1 << 0);
    
    // Set PA5 as output  
    volatile unsigned int *moder = (volatile unsigned int*)0x40020000;
    *moder &= ~(3 << 10);
    *moder |= (1 << 10);
    
    // Infinite blink - should be clearly visible
    while(1) {
        *((volatile unsigned int*)0x40020018) = (1 << 5);   // LED ON
        delay();
        *((volatile unsigned int*)0x40020018) = (1 << 21);  // LED OFF  
        delay();
    }
}
ENDFILE

echo "📋 Simple blink program created"

echo "🔨 Compiling simple blink test..."
arm-none-eabi-gcc \
    -mcpu=cortex-m4 -mthumb -Os \
    -nostdlib -nostartfiles \
    -T complete.ld \
    simple_blink.c -o simple_blink.elf

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    
    arm-none-eabi-objcopy -O binary simple_blink.elf simple_blink.bin
    
    echo "📊 Binary size:"
    ls -lh simple_blink.bin
    
    echo "📱 Flashing simple blink test..."
    st-flash write simple_blink.bin 0x08000000
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🔍 SIMPLE BLINK TEST DEPLOYED!"
        echo "============================="
        echo ""
        echo "👀 Expected behavior:"
        echo "   - Green LED blinks continuously"
        echo "   - 1 second ON, 1 second OFF"
        echo "   - Should start immediately"
        echo ""
        echo "🔄 If NO blinking:"
        echo "   - Press black RESET button once"
        echo "   - Check red power LED is ON"
        echo ""
        echo "📋 Results:"
        echo "   ✅ LED blinks = STM32 hardware working"
        echo "   ❌ No blinks = Hardware/power issue"
        echo ""
        echo "🎯 This test isolates hardware vs software problems"
    else
        echo "❌ Flash failed!"
    fi
else
    echo "❌ Compilation failed!"
fi