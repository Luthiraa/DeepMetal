#include "../output/model.h"

void Reset_Handler(void);
void delay(volatile unsigned int count);
void setup_led(void);
void setup_uart(void);
void uart_send_string(const char* str);
void uart_send_number(int num);

// Test MNIST patterns (3 different test cases)
static const float test_patterns[3][784] = {
    {[0 ... 783] = 0.0f},  // Pattern 0: all zeros
    {[0 ... 783] = 0.0f, [100] = 0.8f, [128] = 0.8f, [156] = 0.8f, [184] = 0.8f},  // Pattern 1: vertical line
    {[0 ... 783] = 0.0f, [100] = 0.7f, [101] = 0.9f, [102] = 0.7f, [130] = 0.8f, [158] = 0.9f}  // Pattern 2: curve
};

// Vector table
__attribute__((section(".isr_vector")))
void (* const vector_table[])(void) = {
    (void (*)(void))0x20020000,
    Reset_Handler,
};

void delay(volatile unsigned int count) { 
    while(count--); 
}

void setup_led(void) {
    *((volatile unsigned int*)0x40023830) |= (1 << 0);
    volatile unsigned int *moder = (volatile unsigned int*)0x40020000;
    *moder &= ~(3 << 10);
    *moder |= (1 << 10);
}

void led_blink(void) {
    *((volatile unsigned int*)0x40020018) = (1 << 5);
    delay(300000);
    *((volatile unsigned int*)0x40020018) = (1 << 21);
    delay(300000);
}

void setup_uart(void) {
    *((volatile unsigned int*)0x40023830) |= (1 << 0);
    *((volatile unsigned int*)0x40023840) |= (1 << 17);
    
    volatile unsigned int *moder = (volatile unsigned int*)0x40020000;
    volatile unsigned int *afrl = (volatile unsigned int*)0x40020020;
    
    *moder &= ~(3 << 4);
    *moder |= (2 << 4);
    *afrl &= ~(0xF << 8);
    *afrl |= (7 << 8);
    
    volatile unsigned int *brr = (volatile unsigned int*)0x40004408;
    volatile unsigned int *cr1 = (volatile unsigned int*)0x4000440C;
    
    *brr = 208;  // 38400 baud (tested and working!)
    *cr1 = (1 << 13) | (1 << 3);
}

void uart_send_string(const char* str) {
    volatile unsigned int *sr = (volatile unsigned int*)0x40004400;
    volatile unsigned int *dr = (volatile unsigned int*)0x40004404;
    
    while(*str) {
        while(!((*sr) & (1 << 7)));
        *dr = *str++;
    }
}

void uart_send_number(int num) {
    if(num < 0) {
        uart_send_string("-");
        num = -num;
    }
    
    if(num == 0) {
        uart_send_string("0");
        return;
    }
    
    char buffer[12];
    int i = 0;
    while(num > 0) {
        buffer[i++] = '0' + (num % 10);
        num /= 10;
    }
    
    while(i > 0) {
        volatile unsigned int *sr = (volatile unsigned int*)0x40004400;
        volatile unsigned int *dr = (volatile unsigned int*)0x40004404;
        while(!((*sr) & (1 << 7)));
        *dr = buffer[--i];
    }
}

void Reset_Handler(void) {
    setup_led();
    setup_uart();
    
    // Startup sequence - 3 blinks
    for(int i = 0; i < 3; i++) {
        led_blink();
    }
    
    // Send startup banner
    uart_send_string("\r\n*** STM32F446RE Neural Network Demo ***\r\n");
    uart_send_string("========================================\r\n");
    uart_send_string("Model: Nano linear network (3K parameters)\r\n");
    uart_send_string("Architecture: 784 -> 4 -> 2 -> 10\r\n");
    uart_send_string("UART: 38400 baud (WORKING!)\r\n");
    uart_send_string("Converted from PyTorch to C\r\n\r\n");
    
    int test_count = 0;
    
    // Main neural network testing loop
    while(1) {
        for(int pattern = 0; pattern < 3; pattern++) {
            uart_send_string("Neural Network Test ");
            uart_send_number(++test_count);
            uart_send_string(" (Pattern ");
            uart_send_number(pattern);
            uart_send_string("): ");
            
            // *** RUN NEURAL NETWORK INFERENCE ***
            int prediction = predict(test_patterns[pattern], 28, 28, 1);
            
            // Clamp prediction to valid range
            if(prediction < 0) prediction = 0;
            if(prediction > 9) prediction = 9;
            
            uart_send_string("Predicted digit ");
            uart_send_number(prediction);
            uart_send_string("\r\n");
            
            // Blink LED for each prediction
            led_blink();
            delay(2000000);  // 2 second delay
        }
        
        uart_send_string("\r\n--- Neural network cycle complete ---\r\n\r\n");
        delay(1000000);  // 1 second pause before next cycle
    }
}
