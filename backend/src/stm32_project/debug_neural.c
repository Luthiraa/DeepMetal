#include "../output/model.h"

// --- enable FPU on Cortex-M4 ---------------------------------
#define SCB_CPACR (*(volatile unsigned int*)0xE000ED88)
static inline void enable_fpu(void) {
    SCB_CPACR |= (0xF << 20);
    __asm volatile("dsb");
    __asm volatile("isb");
}
// ------------------------------------------------------------

void Reset_Handler(void);
void delay(volatile unsigned int count);
void setup_led(void);
void setup_uart(void);
void uart_send_string(const char* str);
void uart_send_char(char c);
void led_blink(int times);

__attribute__((section(".isr_vector")))
void (* const vector_table[])(void) = {
    (void (*)(void))0x20020000,
    Reset_Handler,
};

void delay(volatile unsigned int count) {
    while (count--);
}

void setup_led(void) {
    *((volatile unsigned int*)0x40023830) |= (1 << 0); // GPIOA clock
    volatile unsigned int *moder = (volatile unsigned int*)0x40020000;
    *moder &= ~(3 << 10);
    *moder |= (1 << 10);  // PA5 output
}

void led_blink(int times) {
    for (int i = 0; i < times; i++) {
        *((volatile unsigned int*)0x40020018) = (1 << 5);   // ON
        delay(200000);
        *((volatile unsigned int*)0x40020018) = (1 << 21);  // OFF
        delay(200000);
    }
}

void setup_uart(void) {
    *((volatile unsigned int*)0x40023830) |= (1 << 0);      // GPIOA clock
    *((volatile unsigned int*)0x40023840) |= (1 << 17);     // USART2 clock

    volatile unsigned int *moder = (volatile unsigned int*)0x40020000;
    volatile unsigned int *afrl  = (volatile unsigned int*)0x40020020;

    *moder &= ~(3 << 4);   *moder |= (2 << 4);   // PA2 = AF
    *afrl  &= ~(0xF << 8); *afrl  |= (7 << 8);   // AF7 = USART2

    volatile unsigned int *brr = (volatile unsigned int*)0x40004408;
    volatile unsigned int *cr1 = (volatile unsigned int*)0x4000440C;

    *brr = 416;                   // 38400 baud @16MHz
    *cr1 = (1 << 13) | (1 << 3);  // UE | TE
}

void uart_send_char(char c) {
    volatile unsigned int *sr = (volatile unsigned int*)0x40004400;
    volatile unsigned int *dr = (volatile unsigned int*)0x40004404;
    while (!((*sr) & (1 << 7)));
    *dr = c;
}

void uart_send_string(const char* str) {
    while (*str) uart_send_char(*str++);
}

void Reset_Handler(void) {
    enable_fpu();             // ← must enable FPU before any float ops

    setup_led();
    setup_uart();

    uart_send_char('X');
    led_blink(1);

    uart_send_string("\r\n=== UART OK ===\r\n");
    led_blink(2);

    uart_send_string("Preparing input...\r\n");
    static float input[784];
    for (int i = 0; i < 784; i++) {
        input[i] = (i < 10) ? 0.1f : 0.0f;
    }
    led_blink(3);

    uart_send_string("Calling predict()...\r\n");
    led_blink(4);

    int prediction = predict(input, 28, 28, 1);

    uart_send_string("Returned from predict()\r\n");
    led_blink(5);

    uart_send_string("Prediction: ");
    if (prediction >= 0 && prediction <= 9) {
        uart_send_char('0' + prediction);
        uart_send_string("\r\n");
    } else {
        uart_send_string("ERR\r\n");
    }

    led_blink(10);

    while (1) {
        delay(3000000);
        led_blink(1);
    }
}
