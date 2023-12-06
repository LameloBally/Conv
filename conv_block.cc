/*!
 *  Copyright (c) 2018 by Contributors
 * \file conv_block.cc
 * \brief Matrix-Matrix Multiply HLS design.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "./conv_block.h"


// This module performs matrix multiplication of matrices A and B
// Where A is an (m,n) and B is an (n,o) matrix.
// We assume that B is stored transposed, resulting in a (o,n) shape.



'커널 사이즈 : 3, 패딩 = 1, 스트라이드 = 1'

void conv_block(
		int M,
		int N,
		int IC,
		int OC,
		volatile float *a, 
		volatile float *w,
		volatile float *c) {

	#pragma HLS INTERFACE m_axi port = a depth = 4096 offset = slave bundle = a_port
	#pragma HLS INTERFACE m_axi port = w depth = 4096 offset = slave bundle = b_port
	#pragma HLS INTERFACE m_axi port = c depth = 4096 offset = slave bundle = c_port
	
	#pragma HLS INTERFACE s_axilite port = a bundle = CONTROL_BUS
	#pragma HLS INTERFACE s_axilite port = w bundle = CONTROL_BUS
	#pragma HLS INTERFACE s_axilite port = c bundle = CONTROL_BUS
	
	#pragma HLS INTERFACE s_axilite port = M bundle = CONTROL_BUS
	#pragma HLS INTERFACE s_axilite port = N bundle = CONTROL_BUS
	#pragma HLS INTERFACE s_axilite port = IC bundle = CONTROL_BUS
	#pragma HLS INTERFACE s_axilite port = OC bundle = CONTROL_BUS
	
	#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

	'stride = 1, padding = 1인 경우의 OH, OW' 
  	int OH = (N + 2  - KH)  + 1 ;
	int OW = (M + 2  - KW)  + 1 ;

	int PADDED_M = M + 2;
  	int PADDED_N = N + 2;

	float a_buff[IC][M][N];
  	float w_buff[OC][IC][M][N];
  	float c_buff[OC][OH][OW];

	#pragma HLS ARRAY_PARTITION variable=a_buff dim=2 type=complete
	#pragma HLS ARRAY_PARTITION variable=b_buff dim=0 type=complete
	#pragma HLS ARRAY_PARTITION variable=c_buff dim=2 type=complete

	memcpy(&a_buff[0][0], const_cast<float*>(a), sizeof(float) * IC * M * N);
	memcpy(&w_buff[0][0], const_cast<float*>(w), sizeof(float) * OC * IC * M * N);
  	memcpy(&c_buff[0][0], const_cast<float*>(c), sizeof(float) * OC * OH * OW);


	'0 Padding 영역'
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

	for (int ic = 0; ic < IC; ic++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                a_pad[ic][m + 1][n + 1] = a_buff[ic][m][n];
            }
        }
    }

	'im2 col col_buff라는 activation과 w_matrix를 만듬
	w_matrix의 row는 커널 하나를 의미하고, col_buff의 컬럼은 커널이 보는 input 패치를 의미함'

	int KH = 3;
	int KW = 3;

	float col_buff[IC * KH * KW][OH * OW];

	for (int i = 0; i < IC * KH * KW; ++i) {
    for (int j = 0; j < OH * OW; ++j) {
        col_buff[i][j] = 0.0f;
    }
}

	for (int oc = 0; oc < OC; oc++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                for (int ic = 0; ic < IC; ic++) {
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            int im_row = oh + kh;
                            int im_col = ow + kw;
                            col_buff[ic * KH * KW + kh * KW + kw][oh * OW + ow] = a_pad[ic][im_row][im_col];
                        }
                    }
                }
            }
        }
    }

	float w_matrix[OC][IC * KH * KW];

	for (int oc = 0; oc < OC; ++oc) {
		for (int i = 0; i < IC * KH * KW; ++i) {
			w_matrix[oc][i] = 0.0f;
		}
	}


	for (int oc = 0; oc < OC; oc++) {
		for (int ic = 0; ic < IC; ic++) {
			for (int kh = 0; kh < KH; kh++) {
				for (int kw = 0; kw < KW; kw++) {
					w_matrix[oc][ic * KH * KW + kh * KW + kw] = w_buff[oc][ic][kh][kw];
				}
			}
		}
	}

	'원래는 w_matrix와 col_buff의 곱으로 Mat Mul을 해야하나, row major order이기 때문에
	w_matrix와 col_buff의 transpose의 row끼리의 내적 연산으로 결과를 계산하도록함'

	float col_buff_transposed[OH * OW][IC * KH * KW];

	for (int i = 0; i < OH * OW; ++i) {
		for (int j = 0; j < IC * KH * KW; ++j) {
			col_buff_transposed[i][j] = 0.0f;
		}
	}

	for (int i = 0; i < IC * KH * KW; ++i) {
		for (int j = 0; j < OH * OW; ++j) {
			col_buff_transposed[j][i] = col_buff[i][j];
		}
	}

	float result[OC][OH * OW];

	// Initialize result matrix to zero
	for (int i = 0; i < OC; ++i) {
		for (int j = 0; j < OH * OW; ++j) {
			result[i][j] = 0.0f;
		}
	}

	for (int i = 0; i < OC; ++i) { // Rows of w_matrix
	#pragma HLS PIPELINE II = 2
		for (int j = 0; j < OH * OW; ++j) { // Rows of col_buff_transposed
			for (int k = 0; k < IC * KH * KW; ++k) { // Elements in row of w_matrix and row of col_buff_transposed
				result[i][j] += w_matrix[i][k] * col_buff_transposed[j][k];
			}
		}
	}

	for (int oc = 0; oc < OC; ++oc) {
		for (int oh = 0; oh < OH; ++oh) {
			for (int ow = 0; ow < OW; ++ow) {
				c_buff[oc][oh][ow] = result[oc][oh * OW + ow];
			}
		}
	}

	const float bn_moving_mean = 0;
  	const float bn_moving_var = 2;

  	const float bn_gamma = 0.1;
  	const float bn_beta = 0;
	const float epsilon = 1e-5;  // Small constant for numerical stability

	for (int oc = 0; oc < OC; ++oc) {
	#pragma HLS PIPELINE II = 2
		for (int oh = 0; oh < OH; ++oh) {
			for (int ow = 0; ow < OW; ++ow) {
				c_buff[oc][oh][ow] = bn_gamma * ((c_buff[oc][oh][ow] - bn_moving_mean) / sqrt(bn_moving_var + epsilon)) + bn_beta;
			}
		}
	}

	for (int oc = 0; oc < OC; ++oc) {
		for (int oh = 0; oh < OH; ++oh) {
			for (int ow = 0; ow < OW; ++ow) {
				c_buff[oc][oh][ow] = std::max(0.0f, c_buff[oc][oh][ow]);
			}
		}
	}

	for (int oc = 0; oc < OC; ++oc) {
	#pragma HLS PIPELINE II = 2
		for (int oh = 0; oh < OH; ++oh) {
			for (int ow = 0; ow < OW; ++ow) {
				c_buff[oc][oh][ow] = std::max(0.0f, c_buff[oc][oh][ow]);
			}
		}
	}

	// Store C
  memcpy(const_cast<float*>(c), const_cast<float*>(&c_buff[0][0]), sizeof(float) * OC * OH * OW);
}

