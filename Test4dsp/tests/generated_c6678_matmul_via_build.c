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
void matmul_fp32(float* A, float* B, float* C, int32_t core_mask);
void matmul_fp32(float* A, float* B, float* C, int32_t core_mask) {
  if ((GetCoreNum(core_mask) > 0) && (c6678_get_core_id(core_mask) >= 0)) {
    int32_t ax0_0;
    for (ax0_0 = (c6678_get_core_id(core_mask) * (4 / GetCoreNum(core_mask))); ax0_0 < ((c6678_get_core_id(core_mask) == (GetCoreNum(core_mask) - 1)) ? 4 : ((c6678_get_core_id(core_mask) + 1) * (4 / GetCoreNum(core_mask)))); ++ax0_0) {
      float A_global_l2[32768];
      int32_t ax1_0;
      for (ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
        int32_t ax2_0;
        for (ax2_0 = 0; ax2_0 < 4; ++ax2_0) {
          int32_t cse_v1 = (ax2_0 * 32);
          load_row_major_tile((&(A[0])), (&(A_global_l2[0])), (ax0_0 * 32), cse_v1, 32, 32, 128, 4, "global");
          int32_t cse_v2 = (ax1_0 * 32);
          load_row_major_tile((&(B[0])), (&(A_global_l2[16384])), cse_v1, cse_v2, 32, 32, 128, 4, "global");
          int32_t ax0_1;
          for (ax0_1 = 0; ax0_1 < 32; ++ax0_1) {
            int32_t ax1_1;
            for (ax1_1 = 0; ax1_1 < 32; ++ax1_1) {
              int32_t ax2_1;
              for (ax2_1 = 0; ax2_1 < 32; ++ax2_1) {
                int32_t cse_v3 = ((ax0_0 * 4096) + (ax0_1 * 128));
                int32_t cse_v4 = ((cse_v3 + cse_v2) + ax1_1);
                if ((cse_v1 + ax2_1) == 0) {
                  C[cse_v4] = 0.000000e+00f;
                }
                C[cse_v4] = (C[cse_v4] + (A_global_l2[((cse_v3 + cse_v1) + ax2_1)] * A_global_l2[(((((ax2_0 * 4096) + (ax2_1 * 128)) + cse_v2) + ax1_1) + 16384)]));
              }
            }
          }
        }
      }
    }
  }
  C6678E_SyncN(GetCoreNum(core_mask), c6678_get_core_id(core_mask));
}

