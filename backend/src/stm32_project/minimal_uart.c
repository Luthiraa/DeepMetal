// Minimal STM32F446RE UART + LED test (no C runtime)

// Function declarations
void Reset_Handler(void);
void delay(volatile unsigned int count);
void setup_led(void);
void setup_uart(void);
void uart_send_char(char c);
void uart_send_string(const char* str);

// Vector table
__attribute__((section(".isr_vector")))
void (* const vector_table[])(void) = {
    (void (*)(void))0x20020000,  // Stack pointer (end of RAM)
    Reset_Handler,               // Reset handler
};

// Simple delay
void delay(volatile unsigned int count) {
    while(count--);
}

// Setup LED on PA5
void setup_led(void) {
    // Enable GPIOA clock
    *((volatile unsigned int*)0x40023830) |= (1 << 0);
    
    // Set PA5 as output
    volatile unsigned int *gpioa_moder = (volatile unsigned int*)0x40020000;
    *gpioa_moder &= ~(3 << 10);  // Clear mode bits
    *gpioa_moder |= (1 << 10);   // Set as output
}

// LED control
void led_on(void) {
    *((volatile unsigned int*)0x40020018) = (1 << 5);  // Set PA5
}

void led_off(void) {
    *((volatile unsigned int*)0x40020018) = (1 << 21); // Reset PA5
}

// Setup UART2 on PA2 (TX)
void setup_uart(void) {
    // Enable GPIOA and USART2 clocks
    *((volatile unsigned int*)0x40023830) |= (1 << 0);   // GPIOA
    *((volatile unsigned int*)0x40023840) |= (1 << 17);  // USART2
    
    // Configure PA2 for UART TX
    volatile unsigned int *gpioa_moder = (volatile unsigned int*)0x40020000;
    volatile unsigned int *gpioa_afrl = (volatile unsigned int*)0x40020020;
    
    // Set PA2 to alternate function mode
    *gpioa_moder &= ~(3 << 4);   // Clear PA2 mode
    *gpioa_moder |= (2 << 4);    // Alternate function
    
    // Set alternate function 7 (USART2)
    *gpioa_afrl &= ~(0xF << 8);  // Clear AF2 bits
    *gpioa_afrl |= (7 << 8);     // Set AF7
    
    // Configure USART2
    volatile unsigned int *usart2_brr = (volatile unsigned int*)0x40004408;
    volatile unsigned int *usart2_cr1 = (volatile unsigned int*)0x4000440C;
    
    // Set baud rate: 115200 @ 8MHz HSI
    *usart2_brr = 69;  // 8000000 / 115200 ≈ 69
    
    // Enable USART and transmitter
    *usart2_cr1 = (1 << 13) | (1 << 3);  // UE | TE
}

// Send single character
void uart_send_char(char c) {
    volatile unsigned int *usart2_sr = (volatile unsigned int*)0x40004400;
    volatile unsigned int *usart2_dr = (volatile unsigned int*)0x40004404;
    
    // Wait for transmit data register empty
    while(!((*usart2_sr) & (1 << 7)));
    
    // Send character
    *usart2_dr = c;
}

// Send string
void uart_send_string(const char* str) {
    while(*str) {
        uart_send_char(*str);
        str++;
    }
}

// Main program
void Reset_Handler(void) {
    setup_led();
    setup_uart();
    
    // Startup blink sequence
    for(int i = 0; i < 5; i++) {
        led_on();
        delay(200000);
        led_off();
        delay(200000);
    }
    
    // Send startup message
    uart_send_string("\r\n=== STM32F446RE UART Test ===\r\n");
    uart_send_string("LED: 5 blinks = startup OK\r\n");
    uart_send_string("UART: This message = UART OK\r\n\r\n");
    
    // Main loop
    int counter = 0;
    while(1) {
        // Send test message
        uart_send_string("Test ");
        
        // Send counter (simple conversion)
        if(counter < 10) {
            uart_send_char('0' + counter);
        } else if(counter < 100) {
            uart_send_char('0' + (counter / 10));
            uart_send_char('0' + (counter % 10));
        } else {
            uart_send_char('0' + (counter / 100));
            uart_send_char('0' + ((counter / 10) % 10));
            uart_send_char('0' + (counter % 10));
        }
        
        uart_send_string(" - LED blink\r\n");
        
        // Blink LED
        led_on();
        delay(500000);
        led_off();
        delay(1500000);
        
        counter++;
        if(counter > 999) counter = 0;
    }
}
