#ifndef MODEL_H
#define MODEL_H

#define LAYER0_IN_CH 1
#define LAYER0_OUT_CH 2
#define LAYER0_KH 3
#define LAYER0_KW 3
#define LAYER2_IN_CH 2
#define LAYER2_OUT_CH 4
#define LAYER2_KH 3
#define LAYER2_KW 3
#define LAYER4_IN_CH 4
#define LAYER4_OUT_CH 4
#define LAYER4_KH 3
#define LAYER4_KW 3
#define LAYER6_IN_SIZE 576
#define LAYER6_OUT_SIZE 8
#define LAYER8_IN_SIZE 8
#define LAYER8_OUT_SIZE 10
#define MAX_BUFFER_SIZE 2048
#define NUM_LAYERS 9

extern const float conv_w0[LAYER0_OUT_CH][LAYER0_IN_CH][LAYER0_KH][LAYER0_KW];
extern const float conv_b0[LAYER0_OUT_CH];
extern const float conv_w2[LAYER2_OUT_CH][LAYER2_IN_CH][LAYER2_KH][LAYER2_KW];
extern const float conv_b2[LAYER2_OUT_CH];
extern const float conv_w4[LAYER4_OUT_CH][LAYER4_IN_CH][LAYER4_KH][LAYER4_KW];
extern const float conv_b4[LAYER4_OUT_CH];
extern const float linear_w6[LAYER6_OUT_SIZE][LAYER6_IN_SIZE];
extern const float linear_b6[LAYER6_OUT_SIZE];
extern const float linear_w8[LAYER8_OUT_SIZE][LAYER8_IN_SIZE];
extern const float linear_b8[LAYER8_OUT_SIZE];

int predict(const float *input, int input_h, int input_w, int input_ch);
#endif // MODEL_H