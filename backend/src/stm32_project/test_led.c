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
