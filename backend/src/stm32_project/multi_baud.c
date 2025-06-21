void Reset_Handler(void);
void delay(volatile unsigned int count);
void setup_led(void);
void setup_uart(unsigned int baud_div);
void uart_send_string(const char* str);

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

void led_blink(int times) {
    for(int i = 0; i < times; i++) {
        *((volatile unsigned int*)0x40020018) = (1 << 5);
        delay(400000);
        *((volatile unsigned int*)0x40020018) = (1 << 21);
        delay(400000);
    }
}

void setup_uart(unsigned int baud_div) {
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
    
    *brr = baud_div;
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

void Reset_Handler(void) {
    setup_led();
    
    unsigned int baud_rates[] = {833, 416, 208, 138, 69};
    const char* rate_names[] = {"9600", "19200", "38400", "57600", "115200"};
    
    while(1) {
        for(int i = 0; i < 5; i++) {
            setup_uart(baud_rates[i]);
            
            led_blink(i + 1);
            delay(1000000);
            
            uart_send_string("\r\nBAUD TEST: ");
            uart_send_string(rate_names[i]);
            uart_send_string(" bps\r\n");
            uart_send_string("LED blinked ");
            uart_send_string(i == 0 ? "1" : i == 1 ? "2" : i == 2 ? "3" : i == 3 ? "4" : "5");
            uart_send_string(" times\r\n");
            uart_send_string("If you see this clearly, baud rate is correct!\r\n");
            
            for(int j = 0; j < 3; j++) {
                uart_send_string("Test ");
                uart_send_string(j == 0 ? "A" : j == 1 ? "B" : "C");
                uart_send_string("\r\n");
                delay(1500000);
            }
            
            delay(2000000);
        }
    }
}
