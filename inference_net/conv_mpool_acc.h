#ifndef _CONV_MPOOL_ACC_H_
#define _CONV_MPOOL_ACC_H_

// Accelerator customization
#include "acc_0_config.h"

// Accelerator headers
#include "conv_engine.h"
#include "pool_engine.h"
#include "activation_functions.h"

template <typename T, typename W, typename G>
void conv_mpool_acc(
        T in_buf_0[Tn][IBUF_t][IBUF_t],
        T in_buf_1[Tn][IBUF_t][IBUF_t],
        W w_buf_0[Tn][Tm][WBUF_t][WBUF_t],
        W b_buf_0[Tm],
        G out_buf_0[Tm][OBUF_t][OBUF_t],
        G out_buf_1[Tm][OBUF_t][OBUF_t],
        int param1[16],
        int param2[16]) {

    G out_buf_tmp[Tm][Tr][Tc];
    G out_buf_pool_tmp[Tm][Tr][Tc];

#pragma HLS ARRAY_PARTITION variable=out_buf_tmp complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_buf_pool_tmp complete dim=1

    int w_r_offset = param1[9];
    int w_c_offset = param1[10];
    int r_offset = param1[5];
    int c_offset = param1[6];

    // in & out buffer selection, here only in buffer is changed
    int in_buf_flag = param1[12];
    int out_buf_flag= param1[13];

    // conv computation core, manually instance double buffering
    if (in_buf_flag == 0) {
        conv_engine(in_buf_0, w_buf_0, b_buf_0, out_buf_tmp, param1[0], param1[1], param1[2], param1[3], param1[4],
                    w_r_offset, w_c_offset, r_offset, c_offset, param1[14], param1[15]);
    } else {
        conv_engine(in_buf_1, w_buf_0, b_buf_0, out_buf_tmp, param1[0], param1[1], param1[2], param1[3], param1[4],
                    w_r_offset, w_c_offset, r_offset, c_offset, param1[14], param1[15]);
    }


    if (param1[1] >= param1[7] - Tn) {
/*
        for(int j =0; j < Tr; j++) {
            for(int k=0; k < Tc; k++) {
#pragma HLS PIPELINE
                for(int i=0; i < Tm; i++) {
                    out_buf_tmp[i][j][k] = relu(out_buf_tmp[i][j][k]);
                }
            }
        }
*/
        // Pooling accelerator selection signal param1[11]
        if (param1[11]) {
            pool_engine(out_buf_tmp, out_buf_pool_tmp, param2[0], param2[1], param2[2], param2[3], param2[4], param2[5],
                        param2[6], Tr, Tc);


            int r_offset_1 = 0;
            int c_offset_1 = 0;
            //output_size = (input_size + 2 * pad - kernel_size) / stride + 1
//            int r_out = (Tr + 2 * param2[7] - param2[4]) / param2[0] + 1;
//            int c_out = (Tc + 2 * param2[7] - param2[4]) / param2[0] + 1;
            int r_out = Pool_Tr;
            int c_out = Pool_Tc;
            r_offset_1 = r_offset / Tr * r_out;
            c_offset_1 = c_offset / Tc * c_out;

            for (int j = 0; j < r_out; j++) {
                for (int k = 0; k < c_out; k++) {
#pragma HLS PIPELINE
                    for (int i = 0; i < Tm; i++) {
                        if (out_buf_flag == 0) {
                            out_buf_0[i][j + r_offset_1][k + c_offset_1] = relu(out_buf_pool_tmp[i][j][k]);
                        } else {
                            out_buf_1[i][j + r_offset_1][k + c_offset_1] = relu(out_buf_pool_tmp[i][j][k]);
                        }
                    }
                }
            }
        }
        // without pooling, directly output the processed results
        else {
            for (int j = 0; j < Tr; j++) {
                for (int k = 0; k < Tc; k++) {
#pragma HLS PIPELINE
                    for (int i = 0; i < Tm; i++) {
                        if (out_buf_flag == 0) {
                            out_buf_0[i][j + r_offset][k + c_offset] = relu(out_buf_tmp[i][j][k]);
                        } else {
                            out_buf_1[i][j + r_offset][k + c_offset] = relu(out_buf_tmp[i][j][k]);
                        }
                    }
                }
            }
        }
    }
}

#endif
