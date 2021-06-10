__kernel void gemm_nn (const int M, const int N, const int K, const float ALPHA,
     __global float *restrict A, const int la,
     __global float *restrict B, const int lb,
     __global float *restrict C, const int lc)
{
  int i, j, k;
  float A_PART;

  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      A_PART = ALPHA*A[i * la + k];
      for (j = 0; j < N; ++j) {
        C[i * lc + j] += A_PART * B[k * lb + j];
      }
    }
  }
}
