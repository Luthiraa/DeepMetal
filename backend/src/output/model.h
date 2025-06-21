#ifndef MODEL_H
#define MODEL_H

// Very simple neural network: 784 -> 16 -> 10
#define INPUT_SIZE 784
#define HIDDEN_SIZE 16
#define OUTPUT_SIZE 10

extern const float weights1[HIDDEN_SIZE][INPUT_SIZE];
extern const float bias1[HIDDEN_SIZE];
extern const float weights2[OUTPUT_SIZE][HIDDEN_SIZE];
extern const float bias2[OUTPUT_SIZE];

int predict(const float *input, int input_h, int input_w, int input_ch);

#endif // MODEL_H