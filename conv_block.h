/*!
 *  Copyright (c) 2018 by Contributors
 * \file conv_block.h
 * \brief Convolution Block HLS design.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ap_fixed.h>


#ifndef _CONV_BLOCK_H_
#define _CONV_BLOCK_H_



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


#endif  // _CONV_BLOCK_H
