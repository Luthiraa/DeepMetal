// Complete STM32F446RE Neural Network Program
#include "../output/model.h"

// Test MNIST digit pattern (784 values for 28x28 image)
static const float test_digit[784] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // Add some pattern to make it more like a digit
    [100] = 0.8f, [101] = 0.9f, [102] = 0.8f, [103] = 0.0f, [104] = 0.0f, [105] = 0.0f, [106] = 0.0f, [107] = 0.0f, [108] = 0.0f, [109] = 0.0f, [110] = 0.0f, [111] = 0.0f, [112] = 0.0f, [113] = 0.0f, [114] = 0.0f, [115] = 0.0f, [116] = 0.0f, [117] = 0.0f, [118] = 0.0f, [119] = 0.0f, [120] = 0.0f, [121] = 0.0f, [122] = 0.0f, [123] = 0.0f, [124] = 0.0f, [125] = 0.0f, [126] = 0.0f, [127] = 0.0f,
    [128] = 0.7f, [129] = 0.9f, [130] = 0.7f, [131] = 0.0f, [132] = 0.0f, [133] = 0.0f, [134] = 0.0f, [135] = 0.0f, [136] = 0.0f, [137] = 0.0f, [138] = 0.0f, [139] = 0.0f, [140] = 0.0f, [141] = 0.0f, [142] = 0.0f, [143] = 0.0f, [144] = 0.0f, [145] = 0.0f, [146] = 0.0f, [147] = 0.0f, [148] = 0.0f, [149] = 0.0f, [150] = 0.0f, [151] = 0.0f, [152] = 0.0f, [153] = 0.0f, [154] = 0.0f, [155] = 0.0f,
    // Fill rest with zeros - this is a simplified pattern, in real use you'd load actual MNIST digit data
    [156] = 0.0f, [157] = 0.0f, [158] = 0.0f, [159] = 0.0f, [160] = 0.0f, [161] = 0.0f, [162] = 0.0f, [163] = 0.0f, [164] = 0.0f, [165] = 0.0f, [166] = 0.0f, [167] = 0.0f, [168] = 0.0f, [169] = 0.0f, [170] = 0.0f, [171] = 0.0f, [172] = 0.0f, [173] = 0.0f, [174] = 0.0f, [175] = 0.0f, [176] = 0.0f, [177] = 0.0f, [178] = 0.0f, [179] = 0.0f, [180] = 0.0f, [181] = 0.0f, [182] = 0.0f, [183] = 0.0f,
    // Continue with zeros for remaining elements (total must be 784)
    [200 ... 783] = 0.0f
};

// Function declarations
void Reset_Handler(void);
void delay_ms(int ms);
void setup_gpio(void);
void led_on(void);
void led_off(void);
void led_blink(int times);
void main_program(void);

// Vector table
__attribute__((section(".isr_vector")))
unsigned int vector_table[] = {
    0x20020000,                      // Stack pointer (128KB RAM)
    (unsigned int)Reset_Handler | 1, // Reset handler (thumb mode)
    0,                               // NMI Handler
    0,                               // Hard Fault Handler
    // Add more interrupt vectors if needed
};

// Simple delay function
void delay_ms(int ms) {
    // Approximate delay for 16MHz clock
    for(volatile int i = 0; i < ms * 2000; i++) {
        __asm__("nop");
    }
}

// GPIO setup for LED
void setup_gpio(void) {
    // Enable GPIOA clock (RCC_AHB1ENR register)
    volatile unsigned int *rcc_ahb1enr = (volatile unsigned int*)0x40023830;
    *rcc_ahb1enr |= 0x00000001;  // Enable GPIOA
    
    // Configure PA5 as output (GPIOA_MODER register)
    volatile unsigned int *gpioa_moder = (volatile unsigned int*)0x40020000;
    *gpioa_moder &= ~(0x3 << 10);  // Clear bits 10-11
    *gpioa_moder |= (0x1 << 10);   // Set bit 10 (output mode)
}

// LED control functions
void led_on(void) {
    volatile unsigned int *gpioa_bsrr = (volatile unsigned int*)0x40020018;
    *gpioa_bsrr = (1 << 5);  // Set PA5
}

void led_off(void) {
    volatile unsigned int *gpioa_bsrr = (volatile unsigned int*)0x40020018;
    *gpioa_bsrr = (1 << 21); // Reset PA5
}

// Blink LED specified number of times
void led_blink(int times) {
    for(int i = 0; i < times; i++) {
        led_on();
        delay_ms(300);
        led_off();
        delay_ms(300);
    }
}

// Main program logic
void main_program(void) {
    // Initialize GPIO
    setup_gpio();
    
    // Startup sequence: 3 quick blinks
    for(int i = 0; i < 3; i++) {
        led_on();
        delay_ms(100);
        led_off();
        delay_ms(100);
    }
    
    // Long pause after startup
    delay_ms(2000);
    
    // Main loop: run neural network inference
    while(1) {
        // Run neural network prediction
        int prediction = predict(test_digit, 28, 28, 1);
        
        // Ensure prediction is in valid range (0-9)
        if(prediction < 0) prediction = 0;
        if(prediction > 9) prediction = 9;
        
        // Blink LED: prediction + 1 times
        // So digit 0 = 1 blink, digit 1 = 2 blinks, etc.
        led_blink(prediction + 1);
        
        // Long pause before next prediction
        delay_ms(3000);
    }
}

// Reset handler - entry point after reset
void Reset_Handler(void) {
    // Basic startup sequence
    // (In a full implementation, you'd initialize .data and .bss sections here)
    
    // Call main program
    main_program();
    
    // Should never reach here, but just in case
    while(1) {
        // Infinite loop
    }
}
