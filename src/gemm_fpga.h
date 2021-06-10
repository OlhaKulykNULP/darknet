#ifndef GEMM_FPGA_H
#define GEMM_FPGA_H

void gemm_fpga_init();
void gemm_fpga_deinit();

void gemm_nn_fpga(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);

#endif
