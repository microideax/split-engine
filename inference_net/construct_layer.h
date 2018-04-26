#ifndef _CONSTRUCT_NET_H_
#define _CONSTRUCT_NET_H_

#include <iostream>
#include <ap_fixed.h>

#include "acc_0_config.h"
#include "acc_instance.h"
//#include "conv_engine.h"

using namespace std;


void conv_pool_layer(
    int layer_param[16],
    int layer_param_1[16],
    int acc_conv_param[16],
    int acc_pool_param[16],
    data_type_w *conv_weight_mem_port,
    data_type_w *conv_bias_mem_port,
    data_type_w *temp_out_0_1,
    data_type_w *temp_out_1_1  ){


#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS

#pragma HLS INTERFACE bram port=layer_param
#pragma HLS INTERFACE bram port=layer_param_1
#pragma HLS INTERFACE bram port=acc_conv_param
#pragma HLS INTERFACE bram port=acc_pool_param

#pragma HLS INTERFACE m_axi port=conv_weight_mem_port offset=slave bundle=weight_in depth = 30976
#pragma HLS INTERFACE m_axi port=conv_bias_mem_port   offset=slave bundle=bias_in depth = 64
#pragma HLS INTERFACE m_axi port=temp_out_0_1  offset=inout bundle=in depth = 81920
#pragma HLS INTERFACE m_axi port=temp_out_1_1  offset=inout bundle=out depth = 81920

    data_type_w in_buf_0[Tn][IBUF_t][IBUF_t];
    data_type_w in_buf_1[Tn][IBUF_t][IBUF_t];
    data_type_w w_buf_0[Tn][Tm][WBUF_t][WBUF_t];
    data_type_w b_buf_0[Tm];
    data_type_w out_buf_0[Tm][OBUF_t][OBUF_t];
    data_type_w out_buf_1[Tm][OBUF_t][OBUF_t];

    int acc_call_rounds  = 0;
    for(int r = 0; r < layer_param[5]&&r<IBUF_t; r += IBUF_t) {
        for (int c = 0; c < layer_param[6]&&c<IBUF_t; c += IBUF_t) {
            for (int m = 0; m < layer_param[2]; m += Tm) {
                for (int n = 0; n < layer_param[0]; n += Tn) {

                    // fill in buffer
                    acc_conv_param[1] = n;
                    acc_conv_param[2] = r;
                    acc_conv_param[3] = c;
                    if(acc_conv_param[12] == 0){
                        in_buf_load(in_buf_0, temp_out_0_1, 0, n, r, c, layer_param[7], layer_param[1], layer_param[8], layer_param[3], layer_param[4], layer_param[0]);
                    }else{
                        in_buf_load(in_buf_1, temp_out_0_1, 0, n, r, c, layer_param[7], layer_param[1], layer_param[8], layer_param[3], layer_param[4], layer_param[0]);
                    }
                    b_buf_load(b_buf_0, conv_bias_mem_port, layer_param[11], m);
                    w_buf_load(w_buf_0, conv_weight_mem_port, layer_param[10], n, m, layer_param[1], layer_param[0], layer_param[2]);

#if _DEBUG_OUTPUT_
                    ofstream conv_out;
                    conv_out.open("in_buf_data.txt", ios::app);
                    conv_out <<"conv input: "<< endl;
                    for (int i = 0; i < layer_param[0]; i++) {
                        for (int j = 0; j < layer_param[3]+layer_param[8]*2; j++) {
                            for(int k = 0; k < layer_param[4]+layer_param[8]*2; k++){
                                conv_out << in_buf_0[i][j][k] << " ";
                            }
                            conv_out << endl;
                        }
                        conv_out << endl;
                    }
                    conv_out.close();
#endif

#if _DEBUG_OUTPUT_
                    cout << "conv layer parameters :" << endl;
                    for(int i =0; i<16; i++){cout << layer_param[i] << "  ";} cout << endl;
#endif
                    // compute buffered data
                    for(int r_offset=0; r_offset < (OBUF_t>layer_param[5]?layer_param[5]:OBUF_t); r_offset+=Tr) {
                        for(int c_offset=0; c_offset < (IBUF_t>layer_param[6]?layer_param[6]:IBUF_t);c_offset+=Tc) {
                            acc_call_rounds++;
                            acc_conv_param[5] = r_offset;
                            acc_conv_param[6] = c_offset;

#if _DEBUG_OUTPUT_
                            // FPGA impl parameter load should happen here
                            cout << "acc_conv_param = {" ;
                            for (int i = 0; i < 16; i++) {cout << acc_conv_param[i] << ", ";}
                            cout << "}; " << endl;
                            cout << "acc_pool_param = {" ;
                            for (int i = 0; i < 16; i++) {cout << acc_pool_param[i] << ", ";}
                            cout << "}; " << endl;
#endif
                            // Accelerator core execution
                            conv_mpool_acc_syn(in_buf_0, in_buf_1, w_buf_0, b_buf_0, out_buf_0, out_buf_1, acc_conv_param, acc_pool_param);

#if _DEBUG_OUTPUT_
                            cout << "acc call round = " << acc_call_rounds << endl;
                            printf("acc call round with printf = %d\n", acc_call_rounds);
                            ofstream conv_out;
                            conv_out.open("acc_output.txt", ios::app);
            			    conv_out << "output buf: " << endl;
			                for (int i = 0; i < 6; i++) {
				                for (int j = 0; j < 8; j++) {
				                    for (int k = 0; k < 32; k++) {
					                    conv_out << out_buf_0[i][j][k] << " ";
				                    }
			 	                    conv_out << endl;
			                    }
			                    conv_out << endl;
			                }
			                conv_out.close();
#endif
                        }
                    }

                    // read results out
                    if(acc_conv_param[13] == 0){
                        output_res(out_buf_0, temp_out_1_1, m, r, c, layer_param_1[2], layer_param_1[4], layer_param_1[5], 1);
                    }else{
                        output_res(out_buf_1, temp_out_1_1, m, r, c, layer_param_1[2], layer_param_1[4], layer_param_1[5], 1);
                    }

#if _DEBUG_OUTPUT_
                    conv_out.open("conv_pool_buf_data.txt", ios::app);
                    conv_out <<"conv_pool output: "<< endl;
                    for (int i = 0; i < layer_param[2]; i++) {
                        for (int j = 0; j < layer_param_1[4]; j++) {
                            for(int k = 0; k < layer_param_1[5]; k++){
                                conv_out << *(temp_out_1_1 + i*layer_param_1[4]*layer_param_1[5]+j*layer_param_1[5]+k) << " ";
                            }
                            conv_out << endl;
                        }
                        conv_out << endl;
                    }
                    conv_out.close();
#endif
                }
            }
        }
    }

}

#endif //_CONSTRUCT_NET_H_
