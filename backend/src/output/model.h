#ifndef MODEL_H
#define MODEL_H

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
