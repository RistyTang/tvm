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
void softmax_fp32(float* A, float* Out, int32_t core_mask);
void softmax_fp32(float* A, float* Out, int32_t core_mask) {
  if ((GetCoreNum(core_mask) > 0) && (c6678_get_core_id(core_mask) >= 0)) {
    int32_t ax0;
    for (ax0 = (c6678_get_core_id(core_mask) * (8 / GetCoreNum(core_mask))); ax0 < ((c6678_get_core_id(core_mask) == (GetCoreNum(core_mask) - 1)) ? 8 : ((c6678_get_core_id(core_mask) + 1) * (8 / GetCoreNum(core_mask)))); ++ax0) {
      float T_max[1];
      float T_exp[1024];
      float T_sum[1];
      int32_t cse_v1 = (ax0 * 1024);
      int32_t ax1;
      for (ax1 = 0; ax1 < 1024; ++ax1) {
        if (ax1 == 0) {
          T_max[0] = -3.402823e+38f;
        }
        T_max[0] = max(T_max[0], A[(cse_v1 + ax1)]);
      }
      int32_t ax1_1;
      for (ax1_1 = 0; ax1_1 < 1024; ++ax1_1) {
        T_exp[ax1_1] = expf((A[(cse_v1 + ax1_1)] - T_max[0]));
      }
      int32_t ax1_2;
      for (ax1_2 = 0; ax1_2 < 1024; ++ax1_2) {
        if (ax1_2 == 0) {
          T_sum[0] = 0.000000e+00f;
        }
        T_sum[0] = (T_sum[0] + T_exp[ax1_2]);
      }
      int32_t ax1_3;
      for (ax1_3 = 0; ax1_3 < 1024; ++ax1_3) {
        Out[(cse_v1 + ax1_3)] = (T_exp[ax1_3] / T_sum[0]);
      }
    }
  }
  C6678E_SyncN(GetCoreNum(core_mask), c6678_get_core_id(core_mask));
}

