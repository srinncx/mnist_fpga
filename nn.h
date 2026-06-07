#ifndef NN_H
#define NN_H

#include "ap_fixed.h"
#include "ap_int.h"

typedef ap_fixed<16,6>  fixed_t;
typedef ap_fixed<32,12> acc_t;
typedef ap_uint<8>      pixel_t;

#define INPUT_SIZE  784
#define HIDDEN1      64
#define HIDDEN2      32

void nn_top(
    pixel_t    input[INPUT_SIZE],
    ap_uint<1> *result
);

#endif