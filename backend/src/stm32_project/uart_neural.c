// STM32F446RE Neural Network with UART Output
#include "../output/model.h"

// Test MNIST patterns (different digits to test)
static const float test_patterns[3][784] = {
    // Pattern 0: mostly zeros (should predict digit 0)
    {[0 ... 783] = 0.0f},
    
    // Pattern 1: some vertical line pattern (might predict digit 1)
    {[0 ... 783] = 0.0f, [100] = 0.8f, [128] = 0.8f, [156] = 0.8f, [184] = 0.8f},
    
    // Pattern 2: some curved pattern (might predict digit 2 or other)
    {[0 ... 783] = 0.0f, [100] = 0.7f, [101] = 0.9f, [102] = 0.7f, [130] = 0.8f, [158] = 0.9f}
};

// Forward declarations
void Reset_Handler(void);
void setup_uart(void);
void uart_putc(char c);
void uart_puts(const char* str);
void uart_print_number(int num);
void delay_ms(int ms);

// Simple vector table
__attribute__((section(".isr_vector")))
void (* const vector_table[])(void) = {
    (void (*)(void))0x20020000,  // Stack pointer
    Reset_Handler,               // Reset handler
};

// Simple delay
void delay_ms(int ms) {
    for(volatile int i = 0; i < ms * 8000; i++);
}

// UART setup for STM32F446RE (UART2 on PA2/PA3)
void setup_uart(void) {
    // Enable clocks for GPIOA and USART2
    *((volatile unsigned int*)0x40023830) |= (1 << 0);   // GPIOA clock
    *((volatile unsigned int*)0x40023840) |= (1 << 17);  // USART2 clock
    
    // Configure PA2 (TX) and PA3 (RX) for UART2
    volatile unsigned int *gpioa_moder = (volatile unsigned int*)0x40020000;
    volatile unsigned int *gpioa_afrl = (volatile unsigned int*)0x40020020;
    
    // Set PA2 and PA3 to alternate function mode
    *gpioa_moder &= ~((3 << 4) | (3 << 6));  // Clear PA2 and PA3 mode bits
    *gpioa_moder |= (2 << 4) | (2 << 6);     // Set to alternate function
    
    // Set alternate function 7 (USART2) for PA2 and PA3
    *gpioa_afrl &= ~((0xF << 8) | (0xF << 12));  // Clear AF bits
    *gpioa_afrl |= (7 << 8) | (7 << 12);         // Set AF7
    
    // Configure USART2 for 115200 baud, 8N1
    volatile unsigned int *usart2_brr = (volatile unsigned int*)0x40004408;
    volatile unsigned int *usart2_cr1 = (volatile unsigned int*)0x4000440C;
    
    // Baud rate: 115200 @ 16MHz = 16000000 / 115200 = ~139
    *usart2_brr = 139;
    
    // Enable USART, TX, RX
    *usart2_cr1 = (1 << 13) | (1 << 3) | (1 << 2);  // UE | TE | RE
}

// Send character via UART
void uart_putc(char c) {
    volatile unsigned int *usart2_sr = (volatile unsigned int*)0x40004400;
    volatile unsigned int *usart2_dr = (volatile unsigned int*)0x40004404;
    
    // Wait for TX empty
    while(!((*usart2_sr) & (1 << 7)));
    
    // Send character
    *usart2_dr = c;
}

// Send string via UART
void uart_puts(const char* str) {
    while(*str) {
        uart_putc(*str++);
    }
}

// Print number via UART
void uart_print_number(int num) {
    if(num == 0) {
        uart_putc('0');
        return;
    }
    
    char buffer[12];
    int i = 0;
    int is_negative = 0;
    
    if(num < 0) {
        is_negative = 1;
        num = -num;
    }
    
    while(num > 0) {
        buffer[i++] = '0' + (num % 10);
        num /= 10;
    }
    
    if(is_negative) {
        uart_putc('-');
    }
    
    // Print digits in reverse order
    while(i > 0) {
        uart_putc(buffer[--i]);
    }
}

// Main program
void main_program(void) {
    setup_uart();
    
    // Startup message
    uart_puts("\r\n🚀 STM32F446RE Neural Network Demo\r\n");
    uart_puts("=================================\r\n");
    uart_puts("Model: Nano linear network (3K parameters)\r\n");
    uart_puts("Architecture: 784 -> 4 -> 2 -> 10\r\n\r\n");
    
    int test_count = 0;
    
    while(1) {
        // Test different patterns
        for(int pattern = 0; pattern < 3; pattern++) {
            uart_puts("🧪 Test ");
            uart_print_number(++test_count);
            uart_puts(" (Pattern ");
            uart_print_number(pattern);
            uart_puts("): ");
            
            // Run neural network inference
            int prediction = predict(test_patterns[pattern], 28, 28, 1);
            
            // Clamp prediction
            if(prediction < 0) prediction = 0;
            if(prediction > 9) prediction = 9;
            
            uart_puts("Predicted digit ");
            uart_print_number(prediction);
            uart_puts("\r\n");
            
            delay_ms(2000);  // 2 second delay between tests
        }
        
        uart_puts("\r\n--- Cycle complete, repeating ---\r\n\r\n");
        delay_ms(1000);
    }
}

// Reset handler
void Reset_Handler(void) {
    main_program();
    while(1);
}
