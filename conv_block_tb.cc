/*!
 *  Copyright (c) 2018 by Contributors
 * \file conv_Block.cc
 * \brief +Simulation tests for the matrix-matrix multiply design.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "conv_block.h"


unsigned globalSeed;

int main(void) {
  // A : M*N // weight : OC * IC * KH * KW // C = OC * OH * OW 
  const int M = 8;
  const int N = 8;
  
  const int KH = 3;
  const int KW = 3;

  const int IC = 8;
  const int OC = 4;

  const float bn_moving_mean = 0;
  const float bn_moving_var = 2;

  const float bn_gamma = 0.1;
  const float bn_beta = 0;

  // padding, stride
  const int stride[2] = {1,1};
  const int padding[2] = {1,1};

  // padded input a
  int PADDED_M = M + 2*padding[0];
  int PADDED_N = N + 2*padding[1];

  // Output C size
  int OW = (M + 2 * padding[0] - KW) / stride[0] + 1 ;
  int OH = (N + 2 * padding[1] - KH) / stride[1] + 1 ;


  // Input and output array initialization
  float *a = (float *) malloc(sizeof(float) * 4096); // IC * M * N);
  float *weight = (float *) malloc(sizeof(float) * 4096); // OC * IC * KH * KW);
  float *c = (float *) malloc(sizeof(float) * 4096); // M * O);

  // Test outcome
  bool correct = true;

  // Reference output
  float c_ref[OC][OH][OW];
  float c_ref_bn[OC][OH][OW];
  float c_ref_bn_relu[OC][OH][OW];
  
  // BN parameter
  float moving_mean[OC];
  float mobing_var[OC];
  float eps = 0.00001;
  float gamma[OC];
  float beta[OC];

  // BN config initialization
  for (int oc=0; oc < OC; oc++){
    moving_mean[oc] = bn_moving_mean;
    moving_var[oc] = bn_moving_var;
  }

  for (int oc=0; oc < OC; oc++){
    gamma[oc] = bn_gamma;
    beta[oc] = bn_beta;
  } 

  printf("-----------Init A----------- \n");
  int i = 0;
  for (int ic = 0; ic < IC; ic ++){
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {  
        // a[m * N * IC + m * N + n] = (float)(rand() % 1024 - 512) / 512;
        a[ic * M * N + m * N + n] = (float)(i % 1024 - 512) / 512;
        printf("%f ", a[ic * M * N + m * N + n]);
        i++;
      }
      printf("\n");
    }
    printf("\n");
  }

  printf("-----------Init weight----------- \n");
  int j = 0;
  for (int oc=0; oc < OC; oc++){
    for (int ic = 0; ic < IC; ic++){
      for (int k_h = 0; k_h < KH; k_h++) {
        for (int k_w = 0; k_w < KW; k_w++) {  
          // a[m * N * IC + m * N + n] = (float)(rand() % 1024 - 512) / 512;
          weight[oc * IC * KH * KW + ic * KH * KW + k_h * KW + k_w] = (float)(j % 1024 - 512) / 512;
          printf("%.3f ", weight[oc * IC * KH * KW + ic * KH * KW + k_h * KW + k_w]);
          j++;
        }
        printf("\n");
      }
      printf("\n");
    }
  }

  // C init
  printf("-----------Init C and C_ref-----------\n");
  for(int oc=0; oc<OC; oc++){
    for (int oh = 0; oh < OH; oh++) {
      for (int ow = 0; ow < OW; ow++) {
        c[oc * OH * OW + oh * OW + ow] = 0;
        c_ref[oc][oh][ow] = 0;
      }
    }
  }

  // conv_bn_relu implementation
  conv_block(M, N, O, a, b, c);

  // a_pad init
  // printf("-----------A_PAD init----------- \n");
  float a_pad[IC][PADDED_M][PADDED_N];
  for (int ic=0; ic < IC; ic ++){
    for (int p_m=0; p_m < PADDED_M; p_m++){
      for (int p_n=0; p_n < PADDED_N; p_n++){
        a_pad[ic][p_m][p_n] = 0;
        // printf("%.3f ", a_pad[ic][p_m][p_n]);
      }
      // printf("\n");
    }
    // printf("\n");
  }

  for(int ic = 0; ic < IC ; ic ++){
    for(int m=0; m < M; m ++){
      for(int n=0; n < N; n ++){
        a_pad[ic][m+1*padding[0]][n+1*padding[0]] = a[ic * M * N + m * N + n];
      }
    }
  }

  //Convolution
  for (int oc=0; oc < OC; oc ++){
    for(int ic=0; ic < IC; ic ++){
      for(int oh=0; oh < OH; oh ++){
        for(int ow=0; ow < OW; ow ++){
          for(int kh=0; kh < KH; kh ++){
            for(int kw=0; kw < KW; kw ++){
              c_ref[oc][oh][ow] += weight[oc * IC * KH * KW + ic * KH * KW  + kh * KH  + kw] * a_pad[ic][oh * stride[0] + kh][ow * stride[1] + kw];
            }
          }
        }
      }
    }
  }

    //c_ref : m*o
  printf("-----------C ref Check-----------\n");
  for(int oc = 0 ; oc < OC; oc++){
    for(int m = 0; m < OH; m++){
      for(int o = 0; o < OW; o ++){
        printf("%.3f ", c_ref[oc][m][o]);
      }
      printf("\n");
    }
    printf("\n");
  }


  // BatchNorm operation
  printf("-----------After BN----------- \n");
  for (int oc = 0; oc < OC; oc ++){
    for (int m = 0; m < OH; m++) {
      for (int o = 0; o < OW; o++) {
        c_ref_bn[oc][m][o] = ( (c_ref[oc][m][o] - moving_mean[oc]) / sqrt(moving_var[oc]+eps) ) * gamma[oc] + beta[oc];
        printf("%.3f ", c_ref_bn[oc][m][o]);
      }
      printf("\n");
    }
    printf("\n");
  }


  //ReLU
  printf("-----------After ReLU----------- \n");
  for (int oc = 0; oc < OC; oc ++){
    for (int oh = 0; oh < OH; oh++) {
      for (int ow = 0; ow < OW; ow++) {
        c_ref_bn_relu[oc][oh][ow] = (c_ref_bn[oc][oh][ow] > 0) ? c_ref_bn[oc][oh][ow] : 0;
        if (c_ref_bn_relu[oc][oh][ow] != c[oc * OH * OW + oh * OW + ow]){
          correct = false;
          printf("%f\t%f\n", c_ref_bn_relu[oc][oh][ow], c[oc * OH * OW + oh * OW + ow]);
        } 
        printf("%.3f ", c_ref_bn_relu[oc][oh][ow]);
      }
      printf("\n");
    }
    printf("\n");
  }

  // Free arrays
  free(a);
  free(weight);
  free(c);

  if (correct) {
    printf("Test successful\n");
    return 0;
  } else {
    printf("Test unsuccessful\n");
    return -1;
  }
}
