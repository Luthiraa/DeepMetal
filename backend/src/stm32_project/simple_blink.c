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
