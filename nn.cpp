#include "nn.h"
#include "weights.h"

// ── Weight loading helper ─────────────────────────────────────────────────────
// weights.h stores Q6.10 raw int16_t bit patterns.
// .range() does a bit-exact copy into ap_fixed<16,6>: value = raw / 1024.
static inline fixed_t to_fixed(int16_t raw) {
    #pragma HLS INLINE
    ap_int<16> bits = raw;
    fixed_t    f;
    f.range(15, 0) = bits.range(15, 0);
    return f;
}

void nn_top(
    pixel_t    *input,
    ap_uint<1> *result
) {
    #pragma HLS INTERFACE m_axi     port=input  bundle=MAXI  depth=784
    #pragma HLS INTERFACE s_axilite port=input  bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=result bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // W3, b2, b3 are small and accessed with constant indices in their
    // respective loops — safe to partition completely into registers.
    // b1 and h1 are accessed with loop variable j/i (non-constant index)
    // so complete partition causes HLS 200-914 warnings — use cyclic instead.
    #pragma HLS ARRAY_PARTITION variable=W3 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=b1 cyclic  factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=b2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=b3 complete dim=1

    fixed_t x[INPUT_SIZE];
    fixed_t h1[HIDDEN1];
    fixed_t h2[HIDDEN2];

    // h1 accessed with non-constant index in L2 — cyclic to avoid HLS 200-914
    #pragma HLS ARRAY_PARTITION variable=h1 cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=h2 complete dim=1

    // ── Normalisation ─────────────────────────────────────────────────────────
    // No loop-carried dependency — II=1 is achievable.
    NORM: for (int i = 0; i < INPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        ap_ufixed<18, 8> tmp = input[i];
        x[i] = tmp >> 8;
    }

    // ── Layer 1 : Linear(784→64) + ReLU ──────────────────────────────────────
    // DSP accumulator depth reported by HLS = 5 (Depth=5 in DRC HLS 200-1470).
    // The loop-carried 'sum' cannot retire faster than every 5 cycles.
    // Setting II=5 matches the hardware reality — no violation, no warning.
    L1: for (int j = 0; j < HIDDEN1; j++) {
        #pragma HLS LOOP_FLATTEN off
        acc_t sum = to_fixed(b1[j]);
        L1_MAC: for (int i = 0; i < INPUT_SIZE; i++) {
            #pragma HLS PIPELINE II=5
            sum += acc_t(to_fixed(W1[j * INPUT_SIZE + i]) * x[i]);
        }
        h1[j] = (sum > acc_t(0)) ? fixed_t(sum) : fixed_t(0);
    }

    // ── Layer 2 : Linear(64→32) + ReLU ───────────────────────────────────────
    // Same DSP depth = 5.
    L2: for (int j = 0; j < HIDDEN2; j++) {
        #pragma HLS LOOP_FLATTEN off
        acc_t sum = to_fixed(b2[j]);
        L2_MAC: for (int i = 0; i < HIDDEN1; i++) {
            #pragma HLS PIPELINE II=5
            sum += acc_t(to_fixed(W2[j * HIDDEN1 + i]) * h1[i]);
        }
        h2[j] = (sum > acc_t(0)) ? fixed_t(sum) : fixed_t(0);
    }

    // ── Output : Linear(32→1), threshold at 0 ────────────────────────────────
    // DSP depth = 4 for the output accumulator (reported as Depth=4).
    acc_t out_sum = to_fixed(b3[0]);
    OUT: for (int i = 0; i < HIDDEN2; i++) {
        #pragma HLS PIPELINE II=4
        out_sum += acc_t(to_fixed(W3[i]) * h2[i]);
    }

    *result = (out_sum >= acc_t(0)) ? ap_uint<1>(1) : ap_uint<1>(0);
}