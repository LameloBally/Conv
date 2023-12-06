/*!
 *  Copyright (c) 2018 by Contributors
 * \file conv_block.cc
 * \brief Matrix-Matrix Multiply HLS design.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./gemm.h"


// This module performs matrix multiplication of matrices A and B
// Where A is an (m,n) and B is an (n,o) matrix.
// We assume that B is stored transposed, resulting in a (o,n) shape.
void conv_block(
		int M,
		int N,
		int O,
		volatile float *a,
		volatile float *b,
		volatile float *c) {

	#pragma HLS INTERFACE m_axi port = a depth = 4096 offset = slave bundle = a_port
	#pragma HLS INTERFACE m_axi port = b depth = 4096 offset = slave bundle = b_port
	#pragma HLS INTERFACE m_axi port = c depth = 4096 offset = slave bundle = c_port
	
	#pragma HLS INTERFACE s_axilite port = a bundle = CONTROL_BUS
	#pragma HLS INTERFACE s_axilite port = b bundle = CONTROL_BUS
	#pragma HLS INTERFACE s_axilite port = c bundle = CONTROL_BUS
	
	#pragma HLS INTERFACE s_axilite port = M bundle = CONTROL_BUS
	#pragma HLS INTERFACE s_axilite port = N bundle = CONTROL_BUS
	#pragma HLS INTERFACE s_axilite port = O bundle = CONTROL_BUS
	
	#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

	// your code

}

