// tvm target: {"kind":"c6678","tag":"","keys":["cpu"],"vector_bytes":32,"core_freq_mhz":1250,"l2_size":983040,"l2_base_core0":276889600,"l2_core_stride":16777216,"core_num":8,"smc_base":201326592,"dma_align_bytes":64,"smc_size":8388608,"ddr_base":2147483648,"l1_size":32768,"dma_burst_bytes":64,"ddr_size":2147483648,"dma_max_transfer":2147483647}
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <tistdtypes.h>
#include <inttypes.h>
#include <stdbool.h>
#include <78NE/initial.h>
#include <78NE/DMA.h>
void matmul_fp32(float* A, float* B, float* C);
void matmul_fp32(float* A, float* B, float* C) {
  int32_t i;
  for (i = 0; i < 128; ++i) {
    int32_t j;
    for (j = 0; j < 128; ++j) {
      int32_t k;
      for (k = 0; k < 128; ++k) {
        int32_t cse_v1 = (i * 128);
        int32_t cse_v2 = (cse_v1 + j);
        if (k == 0) {
          C[cse_v2] = 0.000000e+00f;
        }
        C[cse_v2] = (C[cse_v2] + (A[(cse_v1 + k)] * B[((k * 128) + j)]));
      }
    }
  }
}

