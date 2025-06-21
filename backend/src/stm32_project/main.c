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
