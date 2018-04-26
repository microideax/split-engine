//
// Created by yaochen on 4/4/18.
//

#ifndef _POOL_ENGINE_H_
#define _POOL_ENGINE_H_

#include "acc_0_config.h"

template <typename T, typename G>
void pool_engine(T in_buf[][Tr][Tc], G out_buf[][Tr][Tc], int S, int n, int r, int c, int K, int R, int C, int TR, int TC){
    for(int i=0; i<K; i++){
        for(int j=0; j<K; j++){
//                for(int tr=0; tr+r<R&&(S * tr + i)<TR; tr++){
//                    for(int tc=0; tc+c<C&&(S * tc + j)<TC; tc++){
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


#endif //TEST_DEMO_CONVACC_3_NEW_POOL_ENGINE_H
