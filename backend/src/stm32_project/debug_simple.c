// Simple STM32F446RE UART + LED test
void setup_led(void) {
    *((volatile unsigned int*)0x40023830) |= (1 << 0);   // Enable GPIOA clock
    volatile unsigned int *moder = (volatile unsigned int*)0x40020000;
    *moder &= ~(3 << 10);  // Clear PA5 mode
    *moder |= (1 << 10);   // Set PA5 as output
}

void led_on(void) {
    *((volatile unsigned int*)0x40020018) = (1 << 5);
}

void led_off(void) {
    *((volatile unsigned int*)0x40020018) = (1 << 21);
}

void delay_ms(int ms) {
    for(volatile int i = 0; i < ms * 1000; i++);
}

void setup_uart(void) {
    // Enable clocks
    *((volatile unsigned int*)0x40023830) |= (1 << 0);   // GPIOA
    *((volatile unsigned int*)0x40023840) |= (1 << 17);  // USART2
    
    // Configure PA2 (TX) for UART
    volatile unsigned int *moder = (volatile unsigned int*)0x40020000;
    volatile unsigned int *afrl = (volatile unsigned int*)0x40020020;
    
    *moder &= ~(3 << 4);   // Clear PA2 mode
    *moder |= (2 << 4);    // Alternate function
    
    *afrl &= ~(0xF << 8);  // Clear AF
    *afrl |= (7 << 8);     // AF7 (USART2)
    
    // UART config
    volatile unsigned int *brr = (volatile unsigned int*)0x40004408;
    volatile unsigned int *cr1 = (volatile unsigned int*)0x4000440C;
    
    *brr = 69;    // 115200 @ 8MHz
    *cr1 = (1 << 13) | (1 << 3);  // Enable UART + TX
}

void uart_putc(char c) {
    volatile unsigned int *sr = (volatile unsigned int*)0x40004400;
    volatile unsigned int *dr = (volatile unsigned int*)0x40004404;
    
    while(!((*sr) & (1 << 7)));  // Wait for TX empty
    *dr = c;
}

void uart_puts(const char* str) {
    while(*str) uart_putc(*str++);
}

// Reset handler function
void Reset_Handler(void);

// Fixed vector table
__attribute__((section(".isr_vector")))
void (* const vector_table[])(void) = {
    (void (*)(void))0x20020000,  // Stack pointer
    Reset_Handler,               // Reset handler
};

void Reset_Handler(void) {
    setup_led();
    setup_uart();
    
    // Startup sequence - 5 quick blinks
    for(int i = 0; i < 5; i++) {
        led_on();
        delay_ms(100);
        led_off();
        delay_ms(100);
    }
    
    delay_ms(1000);  // Pause
    
    int counter = 0;
    while(1) {
        // Send test message
        uart_puts("STM32 Test ");
        
        // Send counter
        if(counter < 10) {
            uart_putc('0' + counter);
        } else {
            uart_putc('0' + (counter / 10));
            uart_putc('0' + (counter % 10));
        }
        uart_puts("\r\n");
        
        // Blink LED
        led_on();
        delay_ms(200);
        led_off();
        delay_ms(800);
        
        counter++;
        if(counter > 99) counter = 0;
    }
}

// Main function (not used in this bare metal setup)
int main(void) {
    // This won't be called - Reset_Handler is the entry point
    while(1);
}
