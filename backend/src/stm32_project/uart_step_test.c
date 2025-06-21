void Reset_Handler(void);
void delay(volatile unsigned int count);

__attribute__((section(".isr_vector")))
void (* const vector_table[])(void) = {
    (void (*)(void))0x20020000,
    Reset_Handler,
};

void delay(volatile unsigned int count) { 
    while(count--); 
}

void setup_led(void) {
    *((volatile unsigned int*)0x40023830) |= (1 << 0);  // enable GPIOA clock
    volatile unsigned int *moder = (volatile unsigned int*)0x40020000;
    *moder &= ~(3 << 10);  // clear PA5
    *moder |=  (1 << 10);  // set PA5 as output
}

void led_blink(int times) {
    for (int i = 0; i < times; i++) {
        *((volatile unsigned int*)0x40020018) = (1 << 5);   // LED ON
        delay(400000);
        *((volatile unsigned int*)0x40020018) = (1 << 21);  // LED OFF
        delay(400000);
    }
}

void setup_uart(void) {
    *((volatile unsigned int*)0x40023830) |= (1 << 0);      // GPIOA clock
    *((volatile unsigned int*)0x40023840) |= (1 << 17);     // USART2 clock

    volatile unsigned int *moder = (volatile unsigned int*)0x40020000;
    volatile unsigned int *afrl = (volatile unsigned int*)0x40020020;

    *moder &= ~(3 << 4);       // clear PA2
    *moder |=  (2 << 4);       // alt func mode
    *afrl  &= ~(0xF << 8);     // clear AF for PA2
    *afrl  |=  (7 << 8);       // AF7 = USART2

    volatile unsigned int *brr = (volatile unsigned int*)0x40004408;
    volatile unsigned int *cr1 = (volatile unsigned int*)0x4000440C;

    *brr = 416;  // 38400 baud assuming 8 MHz clock
    *cr1 = (1 << 13) | (1 << 3);  // UE + TE
}

void uart_send_string(const char* str) {
    volatile unsigned int *sr = (volatile unsigned int*)0x40004400;
    volatile unsigned int *dr = (volatile unsigned int*)0x40004404;

    while (*str) {
        while (!((*sr) & (1 << 7)));  // wait for TXE
        *dr = *str++;
    }
}

void Reset_Handler(void) {
    setup_led();

    led_blink(5);               // Step 1: startup
    delay(2000000);

    led_blink(2);               // Step 2: before UART setup
    setup_uart();

    led_blink(3);               // Step 3: UART setup complete
    delay(1000000);

    led_blink(4);               // Step 4: before sending "X"
    uart_send_string("X");

    led_blink(7);               // Step 5: UART works

    uart_send_string("\r\nUART TEST WORKING!\r\n");

    led_blink(1);               // Step 6: loop
    int counter = 0;
    while (1) {
        uart_send_string("Test ");
        uart_send_string(counter == 0 ? "A" : counter == 1 ? "B" : "C");
        uart_send_string("\r\n");

        led_blink(1);
        delay(3000000);
        counter = (counter + 1) % 3;
    }
}
