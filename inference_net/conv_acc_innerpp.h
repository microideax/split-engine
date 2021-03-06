#ifndef _CONV_ACC_H_
#define _CONV_ACC_H_

#include <iostream>
#include <fstream>
#include "activation_functions.h"
#include "acc_0_config.h"
#include "max_pool_acc_innerpp.h"

#if _C_DEBUG_MODE_
#include <algorithm>
#endif

using namespace std;

template <typename T, typename W, typename G, int Tm, int Tn, int Tr, int Tc, int S_max, int K_max, int IBUF_t, int WBUF_t, int OBUF_t>
class conv_acc {

private:
    int conv_layer_number;

public:
    conv_acc() : conv_layer_number(0) {conv_layer_number = 0;};

    ////------------------------------C++ debugging functions---------------------------------------////
    // Reset output buffer
    void out_buf_reset(G buf[][Tr][Tc]){
        for(int i = 0; i < Tm; i++){
            for(int j = 0; j < Tr; j++){
                for(int k = 0; k < Tc; k++){
                    buf[i][j][k] = G(0);
                }
            }
        }
    }
    // Reset weight buffer
    void w_buf_reset(int K, W buf[][Tm][K_max][K_max]){
        for(int i = 0; i < Tn; i++){
            for(int j = 0; j < Tm; j++){
                for(int k = 0; k < K; k++){
                    for(int l = 0; l < K; l++){
                        buf[i][j][k][l] = W(0);
                    }
                }
            }
        }
    }
    ////-----------------------------Accelerator Functions---------------------------------------////
    // Load bias data
    void b_buf_load(W buf[], W *layer_bias, int bias_offset, int m){
        for(int i = 0; i < Tm; i++){
            buf[i] = *(layer_bias + bias_offset + i + m);
        }
    }

    // Load input data
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
    
    void w_buf_t_load(W buf[][Tm][WBUF_t][WBUF_t], W *layer_weights, int weight_offset, int K, int N, int M, int w_r_offset, int w_c_offset){
        for (int m = 0; m < M; m += Tm) {
            for (int n = 0; n < N; n += Tn) {
                w_c_offset += K*(n/Tn);
                w_r_offset += K*(m/Tm);
                for(int k1 = 0; k1 < K; k1++){
                    for(int k2 = 0; k2 < K; k2++){
                        for(int j = 0; j < Tn && j < N; j++){
                            for(int i = 0; i < Tm && i < M; i++){
#pragma HLS PIPELINE
                                buf[j][i][k1+w_r_offset][k2+w_c_offset] = *(layer_weights + weight_offset + (i+m)*N*K*K + (j+n)*K*K + k1*K + k2);
                           }
                        }
                    }
                }
            }
        }
    }
    // Convolution computation kernel
    void conv_engine(T in_buf[][IBUF_t][IBUF_t], W w_buf[][Tm][WBUF_t][WBUF_t], W b_buf[], G out_buf[][Tr][Tc],
                     int S, int n, int r, int c, int K, int w_r_offset, int w_c_offset, int r_offset, int c_offset, int R, int C){
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
    // Max pooling computation kernel
    void pool_engine(T in_buf[][Tr][Tc], G out_buf[][Tr][Tc], int S, int n, int r, int c, int K, int R, int C, int TR, int TC){
        for(int i=0; i<K; i++){
            for(int j=0; j<K; j++){
                for(int tr=0; tr < Pool_Tr; tr++){
                    for(int tc=0; tc < Pool_Tc; tc++){
#pragma HLS PIPELINE
                        for(int tn=0; tn < Tm; tn++){
#pragma HLS UNROLL
                            out_buf[tn][tr][tc] = (i==0&&j==0)?in_buf[tn][S*tr][S*tc]:((out_buf[tn][tr][tc]>in_buf[tn][S*tr+i][S*tc+j])?out_buf[tn][tr][tc]:in_buf[tn][S*tr+i][S*tc+j]);
                        }
                    }
                }
            }
        }
    }

    // Ouput out_buf data to output interface
    void output_res(G out_buf[][OBUF_t][OBUF_t], G *out_data_1, int out_offset, int n, int m, int r, int c, int N, int M, int R_OUT, int C_OUT, bool act){
        if (n >= N - Tn) {
            for (int j = r; j < r + OBUF_t && j < R_OUT; j++) {
                for (int k = c; k < c + OBUF_t && k < C_OUT; k++) {
                    for (int i = 0; i < Tm && i < M-m; i += 1) {
#pragma HLS PIPELINE
                        if (act) {
                            if (i + 0 < M-m)
                                *(out_data_1 + out_offset + ((i+m)/1) * R_OUT * C_OUT + j * C_OUT + k) = relu(out_buf[i + 0][j - r][k - c]);
                        }
                        else {
                            if (i + 0 < M-m)
                                *(out_data_1 + out_offset + ((i+m)/1) * R_OUT * C_OUT + j * C_OUT + k) = out_buf[i + 0][j - r][k - c];
                        }
                    }
                }
            }
        }
    }

///////////////////////------------------conv accelerator----------------//////////////////////////
#if _LAYER_MODE_
    void conv_layer_acc(
            int N, //input feature number
            int K, //input kernel size
            int M, // output feature number
            int R_IN, // input Row
            int C_IN, // input column
            int R_OUT, // output Row
            int C_OUT,// output column
            int S, // stride size
            int P, // padding size
            bool act, // activation function bit (1-- with act, 0--without act)
            W *layer_weights, //w[M][N][K][K]
            int weight_offset,
            int in_offset,
            int out_offset,
            T *in_data_1, // in_data[N][(R-1)*S + K][(C-1)*S + K] --> [N][(R-1)*S + K - 2*P][(C-1)*S + K - 2*P]
            G *out_data_1){ // out[M][R][C]

        /***************local data buffer******************************/
        T in_buf_1[Tn][(Tr-1)*S_max + K_max][(Tc-1)*S_max + K_max];
        T in_buf_0[Tn][(Tr-1)*S_max + K_max][(Tc-1)*S_max + K_max];
        W w_buf_1[Tn][Tm][K_max][K_max];
        W w_buf_0[Tn][Tm][K_max][K_max];
        G out_buf_0[Tm][Tr][Tc];
        G out_buf_1[Tm][Tr][Tc];

        /***************Ptr and buffer initialization******************************/
        bool in_buf_0_empty = 1;
        bool in_buf_1_empty = 1;
        bool out_buf_0_empty = 1;
        bool out_buf_1_empty = 1;
        int loadbufPtr = 0;
        int combufPtr = 0;
        int resbufPtr = 0;
        bool last_com = 0;
        bool last_load = 0;
        bool last_res = 0;

#if _HLS_MODE_
#pragma HLS ARRAY_PARTITION variable=in_buf_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_buf_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_buf_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_buf_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=w_buf_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_buf_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_buf_1 complete dim=1
#endif

#if _C_DEBUG_MODE_
#if _KERNEL_DEBUG_
            cout << "Starting conv_acc_innerpp layer ...." << endl;
            //buffer local data initiallization: must do it in C++ debug!
            out_buf_reset(out_buf_1);
            out_buf_reset(out_buf_0);
            w_buf_reset(K, w_buf_1);
            w_buf_reset(K, w_buf_0);
#endif
#endif
        for(int r = 0; r < R_OUT; r += Tr){
            for(int c = 0; c < C_OUT; c += Tc){
                for(int m = 0; m < M; m += Tm){
                    for(int n = 0; n < N; n += 2*Tn){
   //--------------------------Load input B W D in ping-pong manner-------------------------//
                        while ((in_buf_0_empty | in_buf_1_empty)&& (!last_load)) {
                            if (loadbufPtr == 1) {
                                cout << "loading input buffer 1...." << endl;
                                // load input data
                                in_buf_load(in_buf_1, in_data_1, in_offset, n+Tn, r, c, S, K, P, R_IN, C_IN, N);
                                // load input weights
                                w_buf_load(w_buf_1, layer_weights, weight_offset, n+Tn, m, K, N, M);
                                in_buf_1_empty = 0;
                                cout << "buffer 1 full" << endl;
                                loadbufPtr = 0;
                                if (n+2*Tn >= N) {last_load = 1;}
                            } else {
                                cout << "loading input buffer 0...." << endl;
                                // load input data
                                in_buf_load(in_buf_0, in_data_1, in_offset, n, r, c, S, K, P, R_IN, C_IN, N);
                                // load input weights
                                w_buf_load(w_buf_0, layer_weights, weight_offset, n, m, K, N, M);
                                in_buf_0_empty = 0;
                                cout << "buffer 0 full" << endl;
                                loadbufPtr = 1;
                                if (n+Tn >= N) {last_load = 1;}
                            }
                       }
                       loadbufPtr = 0;
                       last_load = 0;
   //------------------------------compute buffered data -----------------------------------//
                        while ((!in_buf_0_empty | !in_buf_1_empty)&& (!last_com)) {
                            if (combufPtr == 1) {
                                cout << "computing input buffer 1...." << endl;
                                if(resbufPtr == 1){
                                    conv_engine(in_buf_1, w_buf_1, out_buf_1, S, n+Tn, r, c, K, R_OUT, C_OUT);
                                    out_buf_1_empty = 0;
                                }else{
                                    conv_engine(in_buf_1, w_buf_1, out_buf_0, S, n+Tn, r, c, K, R_OUT, C_OUT);
                                    out_buf_0_empty = 0;
                                }
                                in_buf_1_empty = 1;
                                combufPtr = 0;
                                cout << "buffer 1 computed" << endl;
                                if (n+2*Tn >= N) {last_com = 1;}
                            } else {
                                cout << "computing input buffer 0...." << endl;
                                if(resbufPtr == 1){
                                    conv_engine(in_buf_0, w_buf_0, out_buf_1, S, n, r, c, K, R_OUT, C_OUT);
                                    out_buf_1_empty = 0;
                                }else{
                                    conv_engine(in_buf_0, w_buf_0, out_buf_0, S, n, r, c, K, R_OUT, C_OUT);
                                    out_buf_0_empty = 0;
                                }
                                in_buf_0_empty = 1;
                                combufPtr = 1;
                                cout << "buffer 0 computed" << endl;
                                if (n+Tn >= N) {last_com = 1;}
                            }
                       }
                       combufPtr = 0;
                       last_com = 0;
   //---------------------------transfer output data----------------------------------------//
                        while ((!out_buf_0_empty | !out_buf_1_empty)&& (!last_res)) {
                            if (resbufPtr == 1) {
                                cout << "output buffer 1...." << endl;
                                // transfer output data
                                if (n+Tn >= N) {
                                    last_res = 1;
                                    resbufPtr = 0;
                                    output_res(out_buf_1, out_data_1, out_offset, n, m, r, c, N, M, R_OUT, C_OUT, act);
                                }else if (n+2*Tn >= N) {
                                    last_res = 1;
                                    resbufPtr = 0;
                                    output_res(out_buf_1, out_data_1, out_offset, n+Tn, m, r, c, N, M, R_OUT, C_OUT, act);
                                }
                                out_buf_1_empty = 1;
                                cout << "buffer 1 res" << endl;
                            } else {
                                cout << "output buffer 0...." << endl;
                                // transfer output data
                                if (n+Tn >= N) {
                                    last_res = 1;
                                    resbufPtr = 1;
                                    output_res(out_buf_0, out_data_1, out_offset, n, m, r, c, N, M, R_OUT, C_OUT, act);
                                }else if (n+2*Tn >= N) {
                                    last_res = 1;
                                    resbufPtr = 1;
                                    output_res(out_buf_0, out_data_1, out_offset, n+Tn, m, r, c, N, M, R_OUT, C_OUT, act);
                                }
                                out_buf_0_empty = 1;
                                cout << "buffer 0 res" << endl;
                            }
                        }
                        last_res = 0;
                    }
                }
            }
        }
#if _C_DEBUG_MODE_
#if _KERNEL_DEBUG_
            cout << "Finished conv_acc_innerpp layer ...." << endl;
            ofstream conv_out;
            conv_out.open("conv_out_data.txt",ios::app);
            conv_out <<"conv output: "<< endl;
            for (int i = 0; i < M/1; i++) {
                for (int j = 0; j < R_OUT; j++) {
                    for(int k = 0; k < C_OUT; k++){
                        conv_out << *(out_data_1 + out_offset + i*R_OUT*C_OUT + j*C_OUT + k) << " ";
                    }conv_out << endl;
                }conv_out << endl;
            }conv_out.close();
#endif
#endif
    }
#endif
    
   void conv_core_acc( 
        data_type_w in_buf_0[Tn][IBUF_t][IBUF_t],
        data_type_w in_buf_1[Tn][IBUF_t][IBUF_t],
        data_type_w w_buf_0[Tn][Tm][WBUF_t][WBUF_t],
        data_type_w b_buf_0[Tm],
        data_type_w out_buf_0[Tm][OBUF_t][OBUF_t],
        data_type_w out_buf_1[Tm][OBUF_t][OBUF_t],
        int param1[16],
        int param2[16]) {
    
        data_type_w out_buf_tmp[Tm][Tr][Tc];
        data_type_w out_buf_pool_tmp[Tm][Tr][Tc];
        
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

            for(int j =0; j < Tr; j++) {
                for(int k=0; k < Tc; k++) {
#pragma HLS PIPELINE
                    for(int i=0; i < Tm; i++) {
                        out_buf_tmp[i][j][k] = relu(out_buf_tmp[i][j][k]);
                    }
                }
            }

            pool_engine(out_buf_tmp, out_buf_pool_tmp, param2[0], param2[1], param2[2], param2[3], param2[4], param2[5], param2[6], Tr, Tc);

            int r_offset_1=0;
            int c_offset_1=0;
            //output_size = (input_size + 2 * pad - kernel_size) / stride + 1
//            int r_out = (Tr + 2 * param2[7] - param2[4]) / param2[0] + 1;
//            int c_out = (Tc + 2 * param2[7] - param2[4]) / param2[0] + 1;

            int r_out = Pool_Tr;
            int c_out = Pool_Tc;
            r_offset_1 = r_offset / Tr * r_out;
            c_offset_1 = c_offset / Tc * c_out;

            for(int j =0; j < r_out; j++) {
                for(int k=0; k < c_out; k++) {
#pragma HLS PIPELINE
                    for(int i=0; i < Tm; i++) {
                        if (out_buf_flag == 0){
                            out_buf_0[i][j+r_offset_1][k+c_offset_1] = out_buf_pool_tmp[i][j][k];
                        } else {
                            out_buf_1[i][j+r_offset_1][k+c_offset_1] = out_buf_pool_tmp[i][j][k];
                        }
                    }
                }
            }
        }
    }

};
#endif
  