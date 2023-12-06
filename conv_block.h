/*!
 *  Copyright (c) 2018 by Contributors
 * \file gemm.h
 * \brief Matrix-Matrix Multiply HLS design.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ap_fixed.h>


#ifndef _GEMM_H_
#define _GEMM_H_



#ifndef NO_SIM
void conv_block(
  int M,
  int N,
  int IC,
	int OC,
  volatile float *a,
  volatile float *w,
  volatile float *c);
#endif  // NO_SIM


#endif  // PART2_GEMM_H_
