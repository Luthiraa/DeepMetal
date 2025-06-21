#ifndef MODEL_H
#define MODEL_H

#define LAYER0_IN_SIZE 784
#define LAYER0_OUT_SIZE 8
#define LAYER2_IN_SIZE 8
#define LAYER2_OUT_SIZE 4
#define LAYER4_IN_SIZE 4
#define LAYER4_OUT_SIZE 10
#define MAX_BUFFER_SIZE 8192
#define NUM_LAYERS 5

extern const float linear_w0[LAYER0_OUT_SIZE][LAYER0_IN_SIZE];
extern const float linear_b0[LAYER0_OUT_SIZE];
extern const float linear_w2[LAYER2_OUT_SIZE][LAYER2_IN_SIZE];
extern const float linear_b2[LAYER2_OUT_SIZE];
extern const float linear_w4[LAYER4_OUT_SIZE][LAYER4_IN_SIZE];
extern const float linear_b4[LAYER4_OUT_SIZE];

int predict(const float *input, int input_h, int input_w, int input_ch);

#endif // MODEL_H