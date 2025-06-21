#include <stdio.h>

int test_large_arrays() {
    // this simulates the static buffer allocation
    static float buf1[65536], buf2[65536];  // 512KB total
    printf("large arrays allocated successfully\n");
    buf1[0] = 1.0f;
    buf2[0] = 2.0f;
    return (int)(buf1[0] + buf2[0]);
}

int main() {
    printf("testing large static arrays...\n");
    int result = test_large_arrays();
    printf("result: %d\n", result);
    return 0;
}
