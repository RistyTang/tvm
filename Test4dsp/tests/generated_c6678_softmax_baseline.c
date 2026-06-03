// tvm target: {"kind":"c6678","tag":"","keys":["cpu"],"vector_bytes":32,"core_freq_mhz":1250,"l2_size":1048576,"l2_base_core0":276824064,"l2_core_stride":16777216,"core_num":8,"smc_base":201326592,"dma_align_bytes":64,"smc_size":8388608,"ddr_base":2147483648,"l1_size":32768,"dma_burst_bytes":64,"ddr_size":2147483648,"dma_max_transfer":2147483647}
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <csl/csl_cache.h>
#include <csl_cacheAux.h>
#include <tistdtypes.h>
#include <inttypes.h>
#include <stdbool.h>
#include <78NE/initial.h>
#include <78NE/DMA.h>
#include <c_api.h>
void softmax_fp32(float* A, float* Out);
void softmax_fp32(float* A, float* Out) {
  float T_max[8];
  float T_exp[8192];
  int32_t i;
  for (i = 0; i < 8; ++i) {
    int32_t k;
    for (k = 0; k < 1024; ++k) {
      if (k == 0) {
        T_max[i] = -3.402823e+38f;
      }
      T_max[i] = max(T_max[i], A[((i * 1024) + k)]);
    }
  }
  int32_t i_1;
  for (i_1 = 0; i_1 < 8; ++i_1) {
    int32_t k_1;
    for (k_1 = 0; k_1 < 1024; ++k_1) {
      int32_t cse_v1 = ((i_1 * 1024) + k_1);
      T_exp[cse_v1] = expf((A[cse_v1] - T_max[i_1]));
    }
  }
  int32_t i_2;
  for (i_2 = 0; i_2 < 8; ++i_2) {
    int32_t k_2;
    for (k_2 = 0; k_2 < 1024; ++k_2) {
      if (k_2 == 0) {
        T_max[i_2] = 0.000000e+00f;
      }
      T_max[i_2] = (T_max[i_2] + T_exp[((i_2 * 1024) + k_2)]);
    }
  }
  int32_t i_3;
  for (i_3 = 0; i_3 < 8; ++i_3) {
    int32_t k_3;
    for (k_3 = 0; k_3 < 1024; ++k_3) {
      int32_t cse_v2 = ((i_3 * 1024) + k_3);
      Out[cse_v2] = (T_exp[cse_v2] / T_max[i_3]);
    }
  }
}

