//
// Created by yaochen on 4/4/18.
//

#ifndef _CONV_ENGINE_H_
#define _CONV_ENGINE_H_

#include "acc_0_config.h"
#include "activation_functions.h"

template <typename T, typename W, typename G>
void conv_engine(T in_buf[][IBUF_t][IBUF_t], W w_buf[][Tm][WBUF_t][WBUF_t], W b_buf[], G out_buf[][Tr][Tc],
                 int S, int n, int r, int c, int K, int w_r_offset, int w_c_offset, int r_offset, int c_offset, int R, int C) {
    for(int i=0; i<K; i++){
        for(int j=0; j<K; j++){
            for(int tr=0; tr < Tr; tr++){
                for(int tc=0; tc < Tc; tc++) {
#pragma HLS PIPELINE
                    for (int tm = 0; tm < Tm; tm++) {
#pragma HLS UNROLL
                        for (int tn = 0; tn < Tn; tn++) {
#pragma HLS UNROLL
                            if (i == 0 && j == 0 && tn == 0 && n == 0)
                                out_buf[tm][tr][tc] = b_buf[tm] +
                                                      w_buf[tn][tm][i + w_r_offset][j + w_c_offset] *
                                                      in_buf[tn][S * (tr) + i + r_offset][S * (tc) + j +
                                                                                          c_offset];
                            else
                                out_buf[tm][tr][tc] = out_buf[tm][tr][tc] +
                                                      w_buf[tn][tm][i + w_r_offset][j + w_c_offset] *
                                                      in_buf[tn][S * (tr) + i + r_offset][S * (tc) + j +
                                                                                          c_offset];
                        }
                    }
                }
            }
        }
    }
}

// Load input data
template <typename T>
void in_buf_load(T buf[][IBUF_t][IBUF_t],T *in_data_1, int in_offset, int n, int r, int c, int S, int K, int P, int R_IN, int C_IN, int N) {
    for (int j = r * S - P; j < (r + (IBUF_t>R_IN?R_IN:IBUF_t) - 1) * S + K - P; j++) {
        for (int k = c * S - P; k < (c + (IBUF_t>C_IN?C_IN:IBUF_t) - 1) * S + K - P; k++) {
#pragma HLS PIPELINE
            for (int i = 0; i < Tn; i+=1){
#pragma HLS UNROLL
                if ((n + Tn > N && i + 0 >= N - n ) || j < 0 || j >= R_IN || k < 0 || k >= C_IN) {
                    buf[i + 0][j - r * S + P][k - c * S + P] = T(0);
                } else {
                    buf[i + 0][j - r * S + P][k - c * S + P] = *(in_data_1 + in_offset + (i + n)/1 * R_IN * C_IN + j * C_IN + k);
                }
            }
        }
    }
}

// Load weights to weight buffer
template <typename W>
void w_buf_load(W buf[][Tm][WBUF_t][WBUF_t], W *layer_weights, int weight_offset, int n, int m, int K, int N, int M){
    for(int k1 = 0; k1 < K; k1++){
        for(int k2 = 0; k2 < K; k2++){
            for(int j = 0; j < Tn && j < N - n; j++){
                for(int i = 0; i < Tm && i < M - m; i++){
#pragma HLS PIPELINE
                    buf[j][i][k1][k2] = *(layer_weights + weight_offset + (i+m)*N*K*K + (j+n)*K*K + k1*K + k2);
                }
            }
        }
    }
}

// Load bias data
template <typename W>
void b_buf_load(W buf[], W *layer_bias, int bias_offset, int m){
    for(int i = 0; i < Tm; i++){
        buf[i] = *(layer_bias + bias_offset + i + m);
    }
}


// Output out_buf data to output interface
template <typename G>
void output_res(G out_buf[][OBUF_t][OBUF_t],G *out_data_1, int n, int r, int c, int N, int R, int C, bool act){
    for (int j = r; j < OBUF_t && j < R; j++) {
        for (int k = c; k < OBUF_t && k < C; k++) {
            for (int i = n; i <  n + Tn && i < N; i += 1) {
#pragma HLS PIPELINE
                if (act) {
                    if (i + 0 < N)
                        *(out_data_1 + (i/1) * R * C + j * C + k) = relu(out_buf[i + 0 - n][j - r][k - c]);
                }
                else {
                    if (i + 0 < N)
                        *(out_data_1 + (i/1) * R * C + j * C + k) = out_buf[i + 0 - n][j - r][k - c];
                }
            }
        }
    }
}

#endif //_CONV_ENGINE_H_
