#ifndef MODEL_H
#define MODEL_H

<<<<<<< HEAD
#define LAYER0_SIZE 128
#define LAYER1_SIZE 10
#define LAYER2_SIZE 16
#define LAYER3_SIZE 32
#define NUM_LAYERS 4

extern const float w0[2048][LAYER0_SIZE];
extern const float b0[LAYER0_SIZE];
extern const float w1[128][LAYER1_SIZE];
extern const float b1[LAYER1_SIZE];
extern const float w2[9][LAYER2_SIZE];
extern const float b2[LAYER2_SIZE];
extern const float w3[144][LAYER3_SIZE];
extern const float b3[LAYER3_SIZE];

int predict(const float *input);
#endif  // MODEL_H
=======
#define LAYER0_IN_SIZE 784
#define LAYER0_OUT_SIZE 128
#define LAYER2_IN_SIZE 128
#define LAYER2_OUT_SIZE 64
#define LAYER4_IN_SIZE 64
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
>>>>>>> ee6b6d3 (somewhat working pipeline)
