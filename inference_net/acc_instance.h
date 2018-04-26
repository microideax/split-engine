#ifndef _ACC_INSTANCE_H_
#define _ACC_INSTANCE_H_

#include "acc_0_config.h"
#include "conv_mpool_acc.h"

// include this header to instantiate the corresponding in/out data functions
//#include "conv_acc_innerpp.h"


//conv_acc<data_type, data_type_w, data_type_o, Tm, Tn, Tr, Tc, S_max, K_max, IBUF_t, WBUF_t, OBUF_t> convAcc1;

//max_pool_acc<data_type, data_type_w, data_type_o, Tm, Tr, Tc, S_max, K_max> maxPoolAcc1;

void conv_mpool_acc_syn(data_type_w in_buf_0[Tn][IBUF_t][IBUF_t],
                   data_type_w in_buf_1[Tn][IBUF_t][IBUF_t],
                   data_type_w w_buf_0[Tn][Tm][WBUF_t][WBUF_t],
                   data_type_w b_buf_0[Tm],
                   data_type_w out_buf_0[Tm][OBUF_t][OBUF_t],
                   data_type_w out_buf_1[Tm][OBUF_t][OBUF_t],
                   int conv_param[16],
                   int pool_param[16]) {

#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS

#pragma HLS ARRAY_PARTITION variable=in_buf_0  complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_buf_1  complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_buf_0   complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_buf_0   complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_buf_1 complete dim=1

#pragma HLS RESOURCE variable=in_buf_0  core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=in_buf_1  core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=w_buf_0   core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=out_buf_0 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=out_buf_1 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=conv_param  core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=pool_param  core=RAM_1P_BRAM

#pragma HLS INTERFACE bram port=in_buf_0
#pragma HLS INTERFACE bram port=in_buf_1
#pragma HLS INTERFACE bram port=w_buf_0
#pragma HLS INTERFACE bram port=b_buf_0
#pragma HLS INTERFACE bram port=out_buf_0
#pragma HLS INTERFACE bram port=out_buf_1
#pragma HLS INTERFACE bram port=conv_param
#pragma HLS INTERFACE bram port=pool_param

    int param1[16];
    int param2[16];

    data_type_w b_buf_0_tmp[Tm];
#pragma HLS ARRAY_PARTITION variable=b_buf_0_tmp complete

    for(int i = 0; i<16; i++){
#pragma HLS PIPELINE
        param1[i] = conv_param[i];
    }

    for(int i = 0; i<16; i++){
#pragma HLS PIPELINE
        param2[i] = pool_param[i];
    }

    for(int i = 0; i<16; i++){
#pragma HLS PIPELINE
        b_buf_0_tmp[i] = b_buf_0[i];
    }

    conv_mpool_acc(in_buf_0, in_buf_1, w_buf_0, b_buf_0_tmp, out_buf_0, out_buf_1, param1, param2);
}


/*
void pool_core_syn(data_type_w in_buf[Tm][OBUF_t][OBUF_t],
                   data_type_w out_buf[Tm][OBUF_t][OBUF_t],
                   int pool_param_in[16])
{
#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS

#pragma HLS ARRAY_PARTITION variable=in_buf  complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=1

#pragma HLS RESOURCE variable=in_buf  core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=out_buf core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=pool_param_in  core=RAM_1P_BRAM

#pragma HLS INTERFACE bram port=in_buf
#pragma HLS INTERFACE bram port=out_buf
#pragma HLS INTERFACE bram port=pool_param_in

    int param[16];
    for(int i = 0; i<16; i++) {
#pragma HLS PIPELINE
        param[i] = pool_param_in[i];
    }
    maxPoolAcc1.mpool_core_acc(in_buf, out_buf, param);

}
*/

#endif
