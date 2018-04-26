
#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <cstdlib>
#include <time.h>

//#include <malloc.h>
#include <stdlib.h>

#include <ap_fixed.h>
#include "inference_net/acc_0_config.h"
#include "inference_net/construct_layer.h"
#include "inference_net/image_converter.h"
#include "inference_net/weight_bias_one_dim.h"
#include "inference_net/softmax_one_dim.h"
#include "inference_net/predict_one_dim.h"
#include "inference_net/accuracy_one_dim.h"
#include "inference_net/pow_function.h"
#include "inference_net/resize_image.h"

using namespace std;

const unsigned char * loadfile(const std::string &file, int &size) {
   std::ifstream fs(file.c_str(), std::ios::binary);
   fs.seekg(0, std::ios::end);
   size = fs.tellg();
   char * data = new char[size];
   fs.seekg(0);
   fs.read(data, sizeof(char) * size);
   fs.close();
   return (unsigned char *)data;
}

int main() {

   cout<< "Calculating memory space ... ... ... ..." << endl;

   // data size calculation
   unsigned int conv_weight_size = (30976) * sizeof(data_type_w);
   unsigned int conv_bias_size = (64) * sizeof(data_type_w);
   unsigned int fc_weight_size = (4000) * sizeof(data_type_w);
   unsigned int fc_bias_size = (10) * sizeof(data_type_w);
   unsigned int fc_3_out_size = (10) * sizeof(data_type_o);
   unsigned int out_size_0_1 = (81920) * sizeof(data_type_o);
   unsigned int out_size_1_1 = (20480) * sizeof(data_type_o);

    /*
//    data_type_w in_buf_0[8][15*5+11][15*5 + 11];
    data_type_w in_buf_0[Tn][IBUF_t][IBUF_t];
    data_type_w in_buf_1[Tn][IBUF_t][IBUF_t];
    data_type_w w_buf_0[Tn][Tm][WBUF_t][WBUF_t];
    data_type_w w_buf_1[Tn][Tm][WBUF_t][WBUF_t];
    data_type_w b_buf_0[Tm];
    data_type_w b_buf_1[Tm];
    data_type_w out_buf_0[Tm][OBUF_t][OBUF_t];
    data_type_w out_buf_1[Tm][OBUF_t][OBUF_t];
*/
   // assign memory space to different ports
   data_type_w *conv_weight_mem_port = (data_type_w*)malloc(conv_weight_size);
   if (conv_weight_mem_port == NULL) {
      printf("False memory allocation of conv_weight_mem_port\n");
   }
   else {
      printf("conv weight memory location= 0x%x \n", conv_weight_mem_port);
   }
   data_type_w *conv_bias_mem_port = (data_type_w*)malloc(conv_bias_size);
   if (conv_bias_mem_port == NULL) {
      printf("False memory allocation of conv_bias_mem_port\n");
   }
   else {
      printf("conv bias memory location= 0x%x \n", conv_bias_mem_port);
   }
   data_type_w *fc_weight_mem_port = (data_type_w*)malloc(fc_weight_size);
   if (fc_weight_mem_port == NULL) {
      printf("False memory allocation of fc_weight_mem_port\n");
   }
   else {
      printf("fc_weight_mem_port memory location= 0x%x \n", fc_weight_mem_port);
   }
   data_type_w *fc_bias_mem_port = (data_type_w*)malloc(fc_bias_size);
   if (fc_bias_mem_port == NULL) {
      printf("False memory allocation of fc_bias_mem_port\n");
   }
   else {
      printf("fc_bias_mem_port memory location= 0x%x \n", fc_bias_mem_port);
   }
   data_type_o *fc_3_out_mem_int = (data_type_o*)malloc(fc_3_out_size);
   if (fc_3_out_mem_int == NULL) {
      printf("False memory allocation of fc_out_mem_int\n");
   }
   else {
      printf("fc_out_mem_int memory location= 0x%x \n", fc_3_out_mem_int);
   }
   data_type_o *temp_out_0_1 = (data_type_o *)malloc(out_size_0_1);
   if (temp_out_0_1 == NULL) {
      printf("False memory allocation of temp_out_0_1\n");
   }
   else {
      printf("temp_out_0_1 memory location= 0x%x \n", temp_out_0_1);
   }
   data_type_o *temp_out_1_1 = (data_type_o *)malloc(out_size_1_1);
   if (temp_out_1_1 == NULL) {
      printf("False memory allocation of temp_out_1_1\n");
   }
   else {
      printf("temp_out_1_1 memory location= 0x%x \n", temp_out_1_1);
   }
#if _KERNEL_DEBUG_
   cout << "FC mem init\n";
   memset(fc_3_out_mem_int, 0, fc_3_out_size);
   memset(conv_weight_mem_port, 0, conv_weight_size); 
   memset(temp_out_0_1, 0, out_size_0_1);
   memset(temp_out_1_1, 0, out_size_1_1);
#endif

   //net weight src *****************************
#if _HLS_MODE_
   const char* weight_src = "net_weights.txt";
#else
   const char* weight_src = "net_inputs/net_weights.txt";
#endif
#if _KERNEL_DEBUG_
#if _HLS_MODE_
   string image_dir = "3.bmp";
#else
   string image_dir = "./net_inputs/test_imgs/3.bmp";
#endif
   int w;
   int h;
   int channels;
   int size;
   const unsigned char * data = loadfile(image_dir, size);
   const unsigned char * image_orig = stbi_load_from_memory(data, size, &w, &h, &channels, 1);
   int in_data_size=0;
   ofstream indata;
   indata.open("in_data.txt");
   for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 28; j++) {
         for (int k = 0; k < 28; k++) {
            indata << image_orig[i *28*28 + 28*j + k] << " ";
         }

         indata << endl;
      }

      indata << endl;
   }
   indata.close();

   cout << "Writing data to input data memory space ... ... ..." << endl;
   for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 28; j++) {
         for (int k = 0; k < 28; k++) {
            temp_out_0_1[in_data_size] = (data_type)image_orig[i*28*28 + 28*j + k];
            //in_buf_0[i][j][k] = (data_type)image_orig[i*28*28 + j*28 + k];
      in_data_size++;
         }

      }

   }
   cout << "Finished writing data to input data memory space ... ..." << endl;
#endif
   char tan_h = 't';
   char relu = 'r';
   char none = 'i';
   int in_number_conv = 0;
   int in_number_fc = 0;
   int in_number_pooling = 0;

   // Prepare weights and bias for conv layer 1
   float *conv_1_weight2D = (float*)malloc(150 * sizeof(float));
   memset(conv_1_weight2D, 0, 150 * sizeof(float));
   load_weight_conv(
weight_src, 
conv_1_weight2D,
 weight_bias_record,
 nn_channel_size_conv, 
 nn_in_number_conv,
 nn_out_number_conv,
 in_number_conv);
   int conv_weight_num=0;
   cout << "Loading conv weight 1 to memory space, starting at: " <<conv_weight_num << '\n';
   for (int i = 0; i < 150; i++) {
      conv_weight_mem_port[conv_weight_num] = (data_type_w)conv_1_weight2D[i];
      conv_weight_num++;
   }
   free(conv_1_weight2D);
   float *conv_1_bias2D = (float*)malloc(6 * sizeof(float));
   memset(conv_1_bias2D, 0, 6 * sizeof(float));
   load_bias_conv(
weight_src, 
conv_1_bias2D,
 weight_bias_record,
 nn_channel_size_conv, 
 nn_in_number_conv,
 nn_out_number_conv,
 in_number_conv);
   int conv_bias_num=0;
   cout << "Loading conv bias 1 to memory space, starting at: " <<conv_bias_num << '\n';
   for (int i = 0; i < 6; i++) {
      conv_bias_mem_port[conv_bias_num] = (data_type_w)conv_1_bias2D[i];
      conv_bias_num++;
   }
   free(conv_1_bias2D);
   in_number_conv++;

   // Prepare weights and bias for conv layer 2
   float *conv_2_weight2D = (float*)malloc(2400 * sizeof(float));
   memset(conv_2_weight2D, 0, 2400 * sizeof(float));
   load_weight_conv(
weight_src, 
conv_2_weight2D,
 weight_bias_record,
 nn_channel_size_conv, 
 nn_in_number_conv,
 nn_out_number_conv,
 in_number_conv);
   cout << "Loading conv weight 2 to memory space, starting at: " <<conv_weight_num << '\n';
   for (int i = 0; i < 2400; i++) {
      conv_weight_mem_port[conv_weight_num] = (data_type_w)conv_2_weight2D[i];
      conv_weight_num++;
   }
   free(conv_2_weight2D);
   float *conv_2_bias2D = (float*)malloc(16 * sizeof(float));
   memset(conv_2_bias2D, 0, 16 * sizeof(float));
   load_bias_conv(
weight_src, 
conv_2_bias2D,
 weight_bias_record,
 nn_channel_size_conv, 
 nn_in_number_conv,
 nn_out_number_conv,
 in_number_conv);
   cout << "Loading conv bias 2 to memory space, starting at: " <<conv_bias_num << '\n';
   for (int i = 0; i < 16; i++) {
      conv_bias_mem_port[conv_bias_num] = (data_type_w)conv_2_bias2D[i];
      conv_bias_num++;
   }
   free(conv_2_bias2D);
   in_number_conv++;

   cout<<"Finished loading conv weight into memory! Total: " <<conv_weight_num  << "... ... ..."<<endl;
   cout<<"Finished loading conv bias into memory! Total: " <<conv_bias_num  << "... ... ..."<<endl;

   // Prepare weights and bias for fc layer 1
   float *fc_1_weight2D = (float*)malloc(4000 * sizeof(float));
   memset(fc_1_weight2D, 0, 4000 * sizeof(float));
   load_weight_fc(
weight_src, 
fc_1_weight2D,
 weight_bias_record,
 nn_channel_size_fc, 
 nn_in_number_fc,
 nn_out_number_fc,
 in_number_fc);
   int fc_weight_num=0;
   cout << "Loading fc weight 1 to memory space, starting at: " <<fc_weight_num << '\n';
   for (int i = 0; i < 4000; i++) {
      fc_weight_mem_port[fc_weight_num] = (data_type_w)fc_1_weight2D[i];
      fc_weight_num++;
   }
   free(fc_1_weight2D);
   float *fc_1_bias2D = (float*)malloc(10 * sizeof(float));
   memset(fc_1_bias2D, 0, 10 * sizeof(float));
   load_bias_fc(
weight_src, 
fc_1_bias2D,
 weight_bias_record,
 nn_channel_size_fc, 
 nn_in_number_fc,
 nn_out_number_fc,
 in_number_fc);
   int fc_bias_num=0;
   cout << "Loading fc bias 1 to memory space, starting at: " <<fc_bias_num << '\n';
   for (int i = 0; i < 10; i++) {
      fc_bias_mem_port[fc_bias_num] = (data_type_w)fc_1_bias2D[i];
      fc_bias_num++;
   }
   free(fc_1_bias2D);
   in_number_fc++;

   cout<<"Finished loading fc weight into memory! Total: " <<fc_weight_num  << "... ... ..."<<endl;
   cout<<"Finished loading fc bias into memory! Total: " <<fc_bias_num  << "... ... ..."<<endl;

#if _KERNEL_DEBUG_
   float fc_3_out[10] = { 0 };
   clock_t start, finish, inf_start, inf_finish;
   double totaltime, inf_time;
   start = clock();
#endif

int dir_control_1[4] = {1, 0, 0, 1};
int dir_control_2[4] = {1, 1, 0, 1};
int dir_control_3[4] = {1, 0, 0, 0};
// param order = {n, k, m, Rin, Cin, Rout, Cout, S, P, act}
int conv_param_1[16] = {1, 5, 6, 28, 28, 28, 28, 1, 2, 1, 0, 0, 0, 0, 1, 1};
int pool_param_1[16] = {28, 28, 6, 2, 14, 14, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0};
int conv_param_2[16] = {6, 5, 16, 14, 14, 10, 10, 1, 0, 1, 150, 6, 0, 0, 1, 1};
int pool_param_2[16] = {10, 10, 16, 2, 5, 5, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0};
int conv_param_3[16] = {16, 5, 10, 5, 5, 1, 1, 5, 0, 1, 0, 0, 0, 0, 1, 1};
int pool_param_3[16] = {10, 10, 10, 2, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0};

int conv_engine_param_in_1[16] = {1/*S*/, 0/*n*/, 0/*r*/, 32/*R*/, 5/*K*/, 28,            28, 1/*N*/,    1, 0, 0, 1, 0, 0, 0/*R_out*/, 0/*C_out*/};
int conv_engine_param_in_2[16] = {1/*S*/, 0/*n*/, 0/*r*/, 0/*c*/,  5/*K*/, 28,            28, 6/*N*/,    1, 0, 0, 1, 0, 0, 0/*R_out*/, 0/*C_out*/};
int fc_engine_param_in[16]     = {5/*S*/, 0/*n*/, 0/*r*/, 0/*c*/,  5/*K*/, 28,            28, 16/*N*/,   1, 0, 0, 0, 0, 0, 1/*R_out*/, 1/*C_out*/};
int pool_engine_param_in_1[16] = {2/*S*/, 0/*n*/, 0/*r*/, 0/*c*/,  2/*K*/, 28/*in_size*/, 28/*in_size*/, 0/*P*/, 0, 0, 0, 0, 0, 0, 0, 0};
int pool_engine_param_in_2[16] = {2/*S*/, 0/*n*/, 0/*r*/, 0/*c*/,  2/*K*/, 10/*in_size*/, 10/*in_size*/, 0/*P*/, 0, 0, 0, 0, 0, 0, 0, 0};
int pool_engine_param_in_3[16] = {1/*S*/, 0/*n*/, 0/*r*/, 0/*c*/,  1/*K*/, 1/*in_size*/,  1/*in_size*/,  0/*P*/, 0, 0, 0, 0, 0, 0, 0, 0};
//    inference_net( dir_control_1, conv_param_1, pool_param_1, conv_weight_mem_port, conv_bias_mem_port, temp_out_0_1, temp_out_1_1);
int w_r_offset = 0;
int w_c_offset = 0;
    /*
#if _ACC_MODE_
    //conv_1_w_load
    convAcc1.w_buf_t_load(w_buf_0, conv_weight_mem_port, conv_param_1[10], conv_param_1[1], conv_param_1[0], conv_param_1[2], w_r_offset, w_c_offset);
    //conv_2_w_load
    w_c_offset = 5;
    conv_engine_param_in_2[10] = w_c_offset;
    convAcc1.w_buf_t_load(w_buf_0, conv_weight_mem_port, conv_param_2[10], conv_param_2[1], conv_param_2[0], conv_param_2[2], w_r_offset, w_c_offset);
    //fc_w_load
    w_c_offset = 10;
    fc_engine_param_in[10] = w_c_offset;
    convAcc1.w_buf_t_load(w_buf_0, fc_weight_mem_port, conv_param_3[10], conv_param_3[1], conv_param_3[0], conv_param_3[2], w_r_offset, w_c_offset);
    ofstream w_buf_t;
    w_buf_t.open("w_buf_data.txt", ios::app);
    w_buf_t <<"w_buf_data: "<< endl;
    for (int i = 0; i < Tn; i++) {
      for (int j = 0; j < Tm; j++) {
        for(int k = 0; k < WBUF_t; k++){
          for(int l = 0; l < WBUF_t; l++){
            w_buf_t << w_buf_0[i][j][k][l] << " ";
          }
          w_buf_t << endl;
        }
        w_buf_t << endl;
      }
      w_buf_t << endl;
    }
    w_buf_t.close();
#endif */

    //conv_1
    conv_pool_layer(
        conv_param_1,
        pool_param_1,
        conv_engine_param_in_1,
        pool_engine_param_in_1,
        conv_weight_mem_port,
        conv_bias_mem_port,
        temp_out_0_1,
        temp_out_1_1
        /*
        in_buf_0,
        in_buf_1,
        w_buf_0,
        b_buf_0,
        out_buf_0,
        out_buf_1,
        w_r_offset,
        w_c_offset
         */
    );

/*
    //conv_2
    conv_pool_layer(
        conv_param_2,
        pool_param_2,
        conv_engine_param_in_2,
        pool_engine_param_in_2,
        conv_weight_mem_port,
        conv_bias_mem_port,
        temp_out_1_1,
        temp_out_0_1);
    //fc_1
    conv_pool_layer(
        conv_param_3,
        pool_param_3,
        fc_engine_param_in,
        pool_engine_param_in_3,
        fc_weight_mem_port,
        fc_bias_mem_port,
        temp_out_0_1,
        temp_out_1_1);
*/

/* Bram interfaced inference_net
inference_net( dir_control_1, conv_param_1, pool_param_1, conv_weight_mem_port, conv_bias_mem_port, temp_out_0_1, temp_out_1_1);
inference_net( dir_control_1, conv_param_2, pool_param_2, conv_weight_mem_port, conv_bias_mem_port, temp_out_0_1, temp_out_1_1);
inference_net( dir_control_3, conv_param_3, pool_param_1,
                   fc_weight_mem_port, fc_bias_mem_port,
                   temp_out_0_1, temp_out_1_1);
*/
/*
   //Original Inference network process
   inference_net(
   //layer weights and bias inputs
   conv_weight_mem_port,
   conv_bias_mem_port,
   fc_weight_mem_port,
   fc_bias_mem_port,
#if _KERNEL_DEBUG_
   //output fc data
   fc_3_out_mem_int,
   temp_out_0_1,
   temp_out_1_1);
*/

#if _KERNEL_DEBUG_
   finish = clock();
   totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
   cout <<"predicted time is: " << totaltime << " s" << endl;
   for (int i = 0; i < 10; i++) {
      fc_3_out[i]=(float)(temp_out_1_1[i]);
   }
   softmax(fc_3_out, 10);
   predict(fc_3_out, 10);
#endif

   return 0;

}
